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

r"""Merge examples.
"""

from absl import app
from absl import flags
import apache_beam as beam
import tensorflow as tf


FLAGS = flags.FLAGS


flags.DEFINE_list("inputs", "", "Input tfrecord files.")

flags.DEFINE_list(
    "multipliers", "", "Optional number of times to duplicate each input."
)

flags.DEFINE_string("output", "", "Output tfrecord file.")


def _get_multipliers():
  if FLAGS.multipliers:
    if len(FLAGS.multipliers) != len(FLAGS.inputs):
      raise ValueError(
          "Invalid multipliers value: %s %s" % (FLAGS.multipliers, FLAGS.inputs)
      )
    return [int(v) for v in FLAGS.multipliers]
  else:
    return [1 for _ in range(len(FLAGS.inputs))]


def pipeline(root):
  """Configure beam pipeline."""
  multipliers = _get_multipliers()
  pcols = []
  input_idx = 1
  for input_path, multiplier in zip(FLAGS.inputs, multipliers):
    print("Reading input `%s` at multipler `%s`." % (input_path, multiplier))
    for _ in range(multiplier):
      pcol = root | f"Read{input_idx}" >> beam.io.ReadFromTFRecord(
          input_path, coder=beam.coders.ProtoCoder(tf.train.Example)
      )
      pcols.append(pcol)
      input_idx += 1

  _ = (
      pcols
      | "Flatten" >> beam.Flatten()
      | "Reshuffle" >> beam.Reshuffle()
      | "Write"
      >> beam.io.WriteToTFRecord(
          FLAGS.output,
          coder=beam.coders.ProtoCoder(tf.train.Example),
      )
  )


def main(argv):
  with beam.Pipeline(
      options=beam.options.pipeline_options.PipelineOptions(argv[1:])) as root:
    pipeline(root)


if __name__ == "__main__":
  app.run(main)
