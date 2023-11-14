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

r"""Run inference for given episode using model server.
"""

import json

from absl import app
from absl import flags
from absl import logging
from pix2act.common import common_utils
from pix2act.common import viz_utils
from pix2act.server import client_utils
from pix2act.tasks.miniwob import miniwob_env
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("miniwob_url", "", "URL for MiniWob server.")

flags.DEFINE_string("server", "", "BNS of server.")

flags.DEFINE_string("subdomain", "use-slider-2", "")

flags.DEFINE_integer("num_seeds", 100, "")

flags.DEFINE_integer("init_seed", 1201539860, "Random seed to start range.")

flags.DEFINE_string("output_dir", "", "Output dir.")

flags.DEFINE_integer("max_steps", 30, "Maximum number of steps.")

flags.DEFINE_bool("break_cycles", True, "Whether to break cycles.")

flags.DEFINE_string(
    "cursor_dir",
    "gs://pix2act-data/cursors/",
    "Directory with cursor files.",
)

flags.DEFINE_bool(
    "output_html",
    True,
    "Writes predictions to HTML files.",
)


def write_json(filepath, data):
  with tf.io.gfile.GFile(filepath, "w") as fp:
    fp.write(json.dumps(data))


def main(unused_argv):
  stub = client_utils.get_stub(FLAGS.server)

  metrics = {
      "task": FLAGS.subdomain,
      "num_episodes": 0,
      "num_complete": 0,
      "num_success": 0,
      "num_failed": 0,
      "total_normalized_reward": 0.0,
      "num_cycles": 0,
      "num_all_actions_filtered": 0,
  }

  env = miniwob_env.MiniWobEnv(
      miniwob_url=FLAGS.miniwob_url,
      cursor_dir=FLAGS.cursor_dir,
  )
  for seed in range(FLAGS.init_seed, FLAGS.init_seed + FLAGS.num_seeds):
    env.start_episode(task=FLAGS.subdomain, seed=seed)
    metrics["num_episodes"] += 1
    steps = []
    step_idx = 1
    cycle_checker = common_utils.CycleChecker()
    while True:
      logging.info("seed: %s, step_idx: %s", seed, step_idx)

      if step_idx >= FLAGS.max_steps:
        break
      step_idx += 1

      metadata = env.get_metadata()
      # Break if episode is complete.
      if metadata["done"]:
        # Write final metadata for debugging purposes.
        step = {}
        step["metadata"] = str(metadata)
        metrics["num_complete"] += 1
        steps.append(step)
        if metadata["raw_reward"] > 0.0:
          metrics["num_success"] += 1
          metrics["total_normalized_reward"] += miniwob_env.normalize_reward(
              metadata["raw_reward"]
          )
        else:
          metrics["num_failed"] += 1
        break

      screenshot_png = env.get_screenshot_png()
      if cycle_checker.is_seen(screenshot_png):
        metrics["num_cycles"] += 1
        if not FLAGS.break_cycles:
          # If we are not breaking cycles or sampling, then this indicates
          # we are stuck in a cycle so break to save time.
          print("Stuck in cycle.")
          break

      beam = client_utils.get_single_beam(stub, screenshot_png)
      predictions = beam.predictions
      if FLAGS.break_cycles:
        predictions = cycle_checker.filter_predictions(
            screenshot_png, predictions
        )
        if not predictions:
          print("No more viable predictions.")
          metrics["num_all_actions_filtered"] += 1
          break
      # Take highest scoring action.
      action_str = predictions[-1].output
      print("action_str: %s" % action_str)
      cycle_checker.add(screenshot_png, action_str)

      # Record step details.
      step = {}
      step["screenshot_png"] = screenshot_png
      step["beam"] = client_utils.beam_to_str(beam)
      step["action_str"] = action_str
      step["metadata"] = str(metadata)
      steps.append(step)
      # Take the step.
      try:
        env.execute_action(action_str)
      except ValueError as e:
        print("Error executing action: %s" % e)
        break
    if FLAGS.output_html:
      html_filepath = f"{FLAGS.output_dir}/{FLAGS.subdomain}-{seed}.html"
      viz_utils.write_steps_to_html(
          steps,
          ["screenshot_png", "beam", "action_str", "metadata"],
          html_filepath,
      )

  metrics_filepath = f"{FLAGS.output_dir}/{FLAGS.subdomain}.json"
  write_json(metrics_filepath, metrics)


if __name__ == "__main__":
  app.run(main)
