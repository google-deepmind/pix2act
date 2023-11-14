# Copyright 2023 The pix2act Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference utils.
"""

import functools
import queue
from typing import Any, Callable, Dict, Iterable, Mapping
from pix2struct import postprocessors
from pix2struct import preprocessors
import seqio
from t5x import models
from t5x import partitioning
from t5x import utils
import tensorflow as tf


OUTPUT_FEATURES = dict(
    inputs=seqio.ContinuousFeature(rank=2, dtype=tf.float32),
    targets=seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(
            "gs://pix2struct-data/sentencepiece.model"
        )
    ),
)

KEY_MAP = dict(
    inputs="image",
    targets="parse",
    parse="parse",
    image="image",
    id="id",
    group_id="group_id",
)

FEATURE_SIGNATURE = {
    "id": tf.TensorSpec(shape=(), dtype=tf.string),
    "image": tf.TensorSpec(shape=(), dtype=tf.string),
    "parse": tf.TensorSpec(shape=(), dtype=tf.string),
    "group_id": tf.TensorSpec(shape=(), dtype=tf.string),
}

# In order to align with the preprocessors used elsewhere we need to change
# the `output_signature` for `parse` to be a sequence, otherwise `sample_one`
# will throw an exception.
PREPROCESSORS = [
    functools.partial(seqio.preprocessors.rekey, key_map=KEY_MAP),
    preprocessors.image_decoder(key="inputs", channels=3),
    preprocessors.normalize_image(key="inputs"),
    preprocessors.image_to_patches(key="inputs"),
    seqio.preprocessors.tokenize_and_append_eos,
]


class InputPipeline:
  """Pipeline for pre-processing the input batches."""

  def __init__(
      self, model, feature_lengths: Mapping[str, int], batch_size: int
  ):
    self._batch_size = batch_size
    self._input_buffer = queue.Queue()

    def _ds_gen():
      while True:
        # No timeout. We want the processing of the input pipeline
        # to wait until the next batch has been fed into queue.
        # Otherwise, it seems TF will try to iterate this generator multiple
        # times perhaps to fill some internal buffer prior to
        # `self._input_iterator` being iterated.
        input_batch = self._input_buffer.get(timeout=None)
        if len(input_batch) != self._batch_size:
          raise ValueError(
              f"Current batch has length {len(input_batch)},"
              f" but expecting batch size of {self._batch_size}"
          )

        for image_bytes in input_batch:
          yield {
              "id": "no-id",
              "group_id": "no-group-id",
              "image": image_bytes,
              "parse": "",
          }

    input_ds = tf.data.Dataset.from_generator(
        _ds_gen, output_signature=FEATURE_SIGNATURE
    )

    self._task = seqio.Task(
        "predict_task",
        source=seqio.FunctionDataSource(
            dataset_fn=lambda split, shuffle_files: input_ds,
            splits=("predict",),
        ),
        preprocessors=PREPROCESSORS,
        output_features=OUTPUT_FEATURES,
        postprocess_fn=postprocessors.multi_target,
    )

    task_ds = self._task.get_dataset(
        sequence_length=feature_lengths,
        split="predict",
        use_cached=False,
        shuffle=False,
    )

    feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)
    self._model_ds = feature_converter(
        task_ds, task_feature_lengths=feature_lengths
    )
    model_ds_batched = self._model_ds.batch(self._batch_size)
    self._input_iterator = model_ds_batched.as_numpy_iterator()

  def feed(self, input_batch):
    if len(input_batch) != self._batch_size:
      raise ValueError(
          f"Current batch has length {len(input_batch)},"
          f" but expecting batch size of {self._batch_size}"
      )
    self._input_buffer.put(input_batch)
    return next(self._input_iterator)

  def input_shapes(self):
    return {
        k: (self._batch_size,) + spec.shape
        for k, spec in self._model_ds.element_spec.items()
    }

  def output_vocab(self):
    return self._task.output_features["targets"].vocabulary


def get_inference_fn(
    num_decodes: int,
    batch_size: int,
    sequence_length: Mapping[str, int],
    model: models.BaseTransformerModel,
    checkpoint_path: str,
    partitioner: partitioning.BasePartitioner,
):
  """Get inference function."""
  input_pipeline = InputPipeline(model, sequence_length, batch_size)
  input_shapes = input_pipeline.input_shapes()
  train_state_initializer = utils.TrainStateInitializer(
      optimizer_def=None,
      init_fn=model.get_initial_variables,
      input_shapes=input_shapes,
      partitioner=partitioner,
  )

  # Restore checkpoint.
  restore_checkpoint_cfg = utils.RestoreCheckpointConfig(
      path=checkpoint_path, mode="specific", strict=False
  )
  train_state = train_state_initializer.from_checkpoint(
      [restore_checkpoint_cfg]
  )
  assert train_state is not None

  # JIT compile the prediction function.
  predict_batch_raw_fn = functools.partial(
      model.predict_batch_with_aux,
      num_decodes=num_decodes,
      return_all_decodes=True,
  )
  predict_batch_fn = partitioner.partition(
      predict_batch_raw_fn,
      in_axis_resources=None,
      out_axis_resources=None,
  )

  vocabulary = input_pipeline.output_vocab()

  def _predict_fn(input_batch):
    model_inputs = input_pipeline.feed(input_batch)
    batched_beam_token_ids, batched_beam_aux = predict_batch_fn(
        train_state.params, model_inputs
    )
    # List of List of (str, float) tuples.
    # TODO(petershaw): Consider avoiding some redundant processing here
    # with populating the model server response.
    batched_beam_out = []
    for beam_token_ids, beam_score in zip(
        batched_beam_token_ids, batched_beam_aux["scores"]
    ):
      beam_out = []
      for token_ids, score in zip(beam_token_ids, beam_score):
        beam_out.append((vocabulary.decode(token_ids), score.item()))
      batched_beam_out.append(beam_out)
    return batched_beam_out

  return _predict_fn
