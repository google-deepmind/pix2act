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

import time

from absl import app
from absl import flags
from pix2act.common import common_utils
from pix2act.common import viz_utils
from pix2act.server import client_utils
from pix2act.tasks.miniwob import miniwob_env


FLAGS = flags.FLAGS

flags.DEFINE_string("miniwob_url", "", "URL of miniwob HTML.")

flags.DEFINE_string("server", "", "BNS of server.")

flags.DEFINE_string("subdomain", "click-test", "")

flags.DEFINE_integer("seed", 0, "")

flags.DEFINE_string("output", "", "Output path for HTML file.")

flags.DEFINE_integer("max_steps", 20, "Maximum number of steps.")

flags.DEFINE_string(
    "cursor_dir",
    "gs://pix2act-data/cursors/",
    "Directory with cursor files.",
)


def main(unused_argv):
  stub = client_utils.get_stub(FLAGS.server)
  env = miniwob_env.MiniWobEnv(
      miniwob_url=FLAGS.miniwob_url,
      cursor_dir=FLAGS.cursor_dir,
  )
  env.start_episode(task=FLAGS.subdomain, seed=FLAGS.seed)
  cycle_checker = common_utils.CycleChecker()
  episode = []

  step_idx = 1
  while True:
    print(f"Step {step_idx}.")
    if step_idx >= FLAGS.max_steps:
      break
    step_idx += 1

    metadata = env.get_metadata()
    if metadata["done"]:
      break

    screenshot_png = env.get_screenshot_png()

    print("Start `get_prediction`")
    start = time.time()
    policy_beam = client_utils.get_single_beam(stub, screenshot_png)
    print("Stop `get_prediction`.")
    elapsed = time.time() - start
    print("Elapsed: %.6f" % elapsed)

    predictions = cycle_checker.filter_predictions(
        screenshot_png, policy_beam.predictions
    )
    action_str = predictions[-1].output
    cycle_checker.add(screenshot_png, action_str)

    # Write steps to HTML.
    step = {}
    step["screenshot_png"] = screenshot_png
    step["metadata"] = str(metadata)
    step["cursor_state"] = str(env.cursor_state)
    step["action_str"] = action_str
    episode.append(step)

    # Take the step.
    try:
      env.execute_action(action_str)
    except ValueError as e:
      print("Error executing action: %s" % e)
      break

  viz_utils.write_steps_to_html(
      episode,
      ["screenshot_png", "cursor_state", "action_str", "metadata"],
      FLAGS.output,
  )


if __name__ == "__main__":
  app.run(main)
