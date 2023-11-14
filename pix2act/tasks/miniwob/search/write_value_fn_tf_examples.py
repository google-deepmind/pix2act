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

r"""Converts episodes to tf examples for training value function approximator.
"""

from absl import app
from absl import flags
import apache_beam as beam
from pix2act.common import tf_utils
from pix2act.tasks.miniwob import episode_pb2
from pix2act.tasks.miniwob.search import reward_utils
import tensorflow as tf


FLAGS = flags.FLAGS


flags.DEFINE_list("inputs", "", "Input tfrecord files of Episodes.")

flags.DEFINE_string("output_dir", "", "Output location for tf examples.")

flags.DEFINE_float(
    "reward_threshold",
    0.8,
    "Demonstrations below this threshold will be discarded.",
)


class ConvertEpisode(beam.DoFn):
  """Convert episode to tf examples."""

  def process(self, episode):
    if not episode.task_name:
      beam.metrics.Metrics.counter("ConvertEpisode", "no_task_name").inc()
    elif not episode.steps:
      beam.metrics.Metrics.counter("no_steps", episode.task_name).inc()
    elif episode.raw_reward < FLAGS.reward_threshold:
      beam.metrics.Metrics.counter(
          "failed_demonstration", episode.task_name
      ).inc()
    else:
      beam.metrics.Metrics.counter("num_demos", episode.task_name).inc()
      try:
        total_steps = len(episode.steps)
        for step_idx, step in enumerate(episode.steps):
          steps_to_go = total_steps - step_idx
          surrogate_reward = reward_utils.compute_surrogate_reward(
              episode.raw_reward, steps_to_go
          )
          value_fn_target = reward_utils.surrogate_reward_to_value_fn_target(
              surrogate_reward
          )
          example = tf.train.Example()

          tf_utils.add_bytes_feature(example, "image", step.screenshot_png)
          tf_utils.add_text_feature(example, "parse", str(value_fn_target))
          tf_utils.add_text_feature(
              example, "id", f"{episode.seed}-{episode.task_name}-{step_idx}"
          )
          yield example
        beam.metrics.Metrics.counter(
            "successful_conversions", episode.task_name
        ).inc()
      except ValueError as _:
        beam.metrics.Metrics.counter(
            "failed_conversions", episode.task_name
        ).inc()
        pass


def pipeline(root):
  """Configure beam pipeline."""

  pcols = []
  for idx, input_path in enumerate(FLAGS.inputs):
    pcol = root | f"Read{idx}" >> beam.io.ReadFromTFRecord(
        input_path,
        coder=beam.coders.ProtoCoder(episode_pb2.Episode),
    )
    pcols.append(pcol)

  _ = (
      pcols
      | "Flatten" >> beam.Flatten()
      | "Reshuffle" >> beam.Reshuffle()
      | "ConvertToTfExamples" >> beam.ParDo(ConvertEpisode())
      | "WriteTfExamples" >> tf_utils.SplitAndWriteTFRecords(FLAGS.output_dir)
  )


def main(argv):
  with beam.Pipeline(
      options=beam.options.pipeline_options.PipelineOptions(argv[1:])) as root:
    pipeline(root)


if __name__ == "__main__":
  app.run(main)
