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

r"""Tool to help visualize episodes.
"""

import os

from absl import app
from absl import flags
from pix2act.common import viz_utils
from pix2act.tasks.miniwob import episode_pb2
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input tfrecord file of Episode protos.")

flags.DEFINE_integer("limit", 10, "Maximum number of episodes to read.")

flags.DEFINE_string("output_dir", "", "Output dir for visualizations.")


def convert_episode(episode):
  """Convert to steps for visualization."""
  steps = []
  for idx, step in enumerate(episode.steps):
    steps.append({
        "screenshot": step.screenshot_png,
        "action": step.action,
        "id": f"{episode.seed}-{episode.task_name}-{idx}",
    })
  return steps


def main(unused_argv):
  raw_dataset = tf.data.TFRecordDataset(FLAGS.input)
  for raw_record in raw_dataset.take(FLAGS.limit):
    episode = episode_pb2.Episode.FromString(raw_record.numpy())
    steps = convert_episode(episode)
    filename = f"{episode.task_name}-{episode.seed}.html"
    filepath = os.path.join(FLAGS.output_dir, filename)
    viz_utils.write_steps_to_html(
        steps, ["id", "screenshot", "action"], filepath
    )


if __name__ == "__main__":
  app.run(main)
