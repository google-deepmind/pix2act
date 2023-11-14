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

r"""Aggregate metrics.
"""

import json

from absl import app
from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "", "Output dir.")


def read_json(filepath):
  with tf.io.gfile.GFile(filepath, "r") as fp:
    return json.load(fp)


def main(unused_argv):
  glob_pattern = f"{FLAGS.output_dir}/*.json"
  task_to_success_rate = {}
  task_to_avg_normalized_reward = {}
  tasks = set()
  for filepath in tf.io.gfile.glob(glob_pattern):
    data = read_json(filepath)
    task = data["task"]
    tasks.add(task)
    success_rate = data["num_success"] / data["num_episodes"]
    task_to_success_rate[task] = success_rate
    avg_normalized_reward = (
        data["total_normalized_reward"] / data["num_episodes"]
    )
    task_to_avg_normalized_reward[task] = avg_normalized_reward

  print("task_name, success_rate, avg_normalized_reward")
  for task in tasks:
    success_rate = task_to_success_rate[task]
    avg_normalized_reward = task_to_avg_normalized_reward[task]
    print("%s, %.2f, %.2f" % (task, success_rate, avg_normalized_reward))


if __name__ == "__main__":
  app.run(main)
