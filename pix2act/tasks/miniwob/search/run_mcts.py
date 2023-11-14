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

r"""Collect episodes using MCTS.
"""

import os
import random

from absl import app
from absl import flags
from absl import logging
from pix2act.server import client_utils
from pix2act.tasks.miniwob import episode_pb2
from pix2act.tasks.miniwob import miniwob_env
from pix2act.tasks.miniwob import miniwob_tasks
from pix2act.tasks.miniwob.search import mcts_utils
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("miniwob_url", "", "URL for MiniWob server.")

flags.DEFINE_string("server_critic", "", "BNS of server for critic.")

flags.DEFINE_string("server_policy", "", "BNS of server for policy.")

flags.DEFINE_string("output_dir", "",
                    "Path to write TFRecord files of Episode protos.")

flags.DEFINE_string("task", "", "Will sample tasks if not set.")

flags.DEFINE_integer("max_episodes", 0, "Run continuously if not set.")

flags.DEFINE_integer("taskid", 0, "Task number (if running multiple tasks)")

flags.DEFINE_bool("raise_exceptions", False, "Whether to raise exceptions")

flags.DEFINE_string(
    "cursor_dir",
    "gs://pix2act-data/cursors/",
    "Directory with cursor files.",
)


EXPLORATION_SCALAR = 0.1
BEAM_LIMIT = 8
MAX_ROLLOUT_DEPTH = 20
NUM_ITERATIONS = 16
MAX_STEPS = 30
ROLLOUT_WEIGHT = 0.9


def _get_mcts_config():
  return mcts_utils.MctsConfig(
      exploration_scalar=EXPLORATION_SCALAR,
      beam_limit=BEAM_LIMIT,
      max_rollout_depth=MAX_ROLLOUT_DEPTH,
      num_iterations=NUM_ITERATIONS,
      max_steps=MAX_STEPS,
      rollout_weight=ROLLOUT_WEIGHT,
  )


def create_episode(step_nodes, task, seed):
  """Creates Episode from `step_nodes`."""
  episode = episode_pb2.Episode(seed=seed, task_name=task)
  final_node = step_nodes[-1]
  episode.raw_reward = final_node.raw_reward()
  episode.complete = final_node.is_complete()

  for node_idx in range(len(step_nodes) - 1):
    parent_node = step_nodes[node_idx]
    child_node = step_nodes[node_idx + 1]
    step = episode.steps.add()
    step.step = node_idx
    step.screenshot_png = parent_node.image()
    step.action = child_node.action_str()

  return episode


def main(unused_argv):
  critic_stub = client_utils.get_stub(FLAGS.server_critic)
  policy_stub = client_utils.get_stub(FLAGS.server_policy)
  env = miniwob_env.MiniWobEnv(FLAGS.miniwob_url, FLAGS.cursor_dir)
  mcts_config = _get_mcts_config()
  max_seed = 2**31
  num_episodes = 0
  output_path = os.path.join(
      FLAGS.output_dir,
      f"episodes-{FLAGS.taskid}.tfrecord",
  )
  with tf.io.TFRecordWriter(output_path) as writer:
    while True:
      if FLAGS.task:
        task = FLAGS.task
      else:
        task = random.choices(list(miniwob_tasks.SUPPORTED_TASKS), k=1)[0]
      seed = random.randint(0, max_seed)

      logging.info("Episode: %s", num_episodes)
      logging.info("Task: %s", task)
      logging.info("Seed: %s", seed)

      try:
        context = mcts_utils.SearchContext(
            mcts_config=mcts_config,
            policy_stub=policy_stub,
            critic_stub=critic_stub,
            env=env,
            seed=seed,
            task=task,
        )
        step_nodes = mcts_utils.run_mcts(context)
        episode = create_episode(step_nodes, task, seed)
        writer.write(episode.SerializeToString())

        # Ensure data is written since we may stop the job arbitrarily.
        writer.flush()

      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error during episode: %s", str(e))
        if FLAGS.raise_exceptions:
          raise e

      num_episodes += 1
      if FLAGS.max_episodes > 0 and num_episodes >= FLAGS.max_episodes:
        break


if __name__ == "__main__":
  app.run(main)
