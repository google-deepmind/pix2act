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

"""Some utilities for reading and writing TF data files."""

import hashlib
import os
from typing import Optional

import apache_beam as beam
import tensorflow as tf


def add_bytes_feature(
    example: tf.train.Example, key: str, value: bytes
) -> None:
  example.features.feature[key].bytes_list.value.append(value)


def add_text_feature(example: tf.train.Example, key: str, value: str) -> None:
  add_bytes_feature(example, key, value.encode("utf-8"))


def get_bytes_feature(example: tf.train.Example, key: str) -> bytes:
  return example.features.feature[key].bytes_list.value[0]


def get_text_feature(example: tf.train.Example, key: str) -> str:
  return get_bytes_feature(example, key).decode("utf-8")


def _get_hash(key: str) -> int:
  return int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)


def _increment_counter(item, counter):
  counter.inc()
  return item


class SplitAndWriteTFRecords(beam.PTransform):
  """Split and write TFRecords."""

  def __init__(
      self,
      output_dir: str,
      validation_percent: Optional[int] = 10,
      train_file_name: str = "train.tfr",
      val_file_name: str = "val.tfr",
  ):
    self._output_dir = output_dir
    self._validation_percent = validation_percent
    self._train_file_name = train_file_name
    self._val_file_name = val_file_name
    self._train_counter = beam.metrics.Metrics.counter(
        "SplitAndWriteTFRecords", "train"
    )
    self._val_counter = beam.metrics.Metrics.counter(
        "SplitAndWriteTFRecords", "val"
    )

  def _partition_index(
      self, example: tf.train.Example, unused_num_partitions: int
  ) -> int:
    key_feature = get_text_feature(example, "id")
    return int(_get_hash(key_feature) % 100 < self._validation_percent)

  def expand(self, pcoll):
    train, val = (
        pcoll
        | "Shuffle" >> beam.Reshuffle()
        | "Partition" >> beam.Partition(self._partition_index, 2)
    )
    _ = (
        train
        | "CountTrain" >> beam.Map(_increment_counter, self._train_counter)
        | "WriteTrain"
        >> beam.io.WriteToTFRecord(
            os.path.join(self._output_dir, self._train_file_name),
            coder=beam.coders.ProtoCoder(tf.train.Example),
        )
    )
    _ = (
        val
        | "CountVal" >> beam.Map(_increment_counter, self._val_counter)
        | "WriteVal"
        >> beam.io.WriteToTFRecord(
            os.path.join(self._output_dir, self._val_file_name),
            coder=beam.coders.ProtoCoder(tf.train.Example),
        )
    )
