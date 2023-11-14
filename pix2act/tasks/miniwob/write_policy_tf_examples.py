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

r"""Aggregates and converts Episode protos to tf examples.
"""

from absl import app
from absl import flags
import apache_beam as beam
from pix2act.common import tf_utils
from pix2act.tasks.miniwob import episode_pb2
import tensorflow as tf


FLAGS = flags.FLAGS


flags.DEFINE_list("input", "", "Path to recordio of OnlineDemo protos.")

flags.DEFINE_string("output_dir", "", "Output location for tf examples.")

flags.DEFINE_float(
    "reward_threshold",
    0.8,
    "Demonstrations below this threshold will be discarded.",
)


def convert_episode_to_tf_examples(episode):
  """Convert online demonstration to training examples."""
  for idx, step in enumerate(episode.steps):
    example = tf.train.Example()
    tf_utils.add_bytes_feature(example, "image", step.screenshot_png)
    tf_utils.add_text_feature(example, "parse", step.action)
    tf_utils.add_text_feature(
        example, "id", f"{episode.seed}-{episode.task_name}-{idx}"
    )
    yield example


class ConvertEpisode(beam.DoFn):
  """Convert demo to tf examples."""

  def process(self, episode):
    if not episode.task_name:
      beam.metrics.Metrics.counter("ConvertEpisode", "no_task_name").inc()
    elif episode.raw_reward < FLAGS.reward_threshold:
      beam.metrics.Metrics.counter(
          "failed_demonstration", episode.task_name
      ).inc()
    else:
      beam.metrics.Metrics.counter("num_demos", episode.task_name).inc()
      try:
        yield from convert_episode_to_tf_examples(episode)
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

  _ = (
      root
      | "Read"
      >> beam.io.ReadFromTFRecord(
          FLAGS.input, coder=beam.coders.ProtoCoder(episode_pb2.Episode)
      )
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
