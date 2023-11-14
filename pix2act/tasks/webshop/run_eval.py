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
import random

from absl import app
from absl import flags
from absl import logging
from pix2act.common import common_utils
from pix2act.common import env_utils
from pix2act.common import render_utils
from pix2act.common import tf_utils
from pix2act.common import viz_utils
from pix2act.server import client_utils
from pix2act.tasks.webshop import demo_utils
from pix2act.tasks.webshop import webshop_env
from selenium import webdriver
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("server", "", "BNS of server.")

flags.DEFINE_string("output_dir", "", "Output dir.")

flags.DEFINE_string("split", "val", "val | test")

flags.DEFINE_integer("max_steps", 30, "Maximum number of steps.")

flags.DEFINE_bool("break_cycles", True, "Whether to break cycles.")

flags.DEFINE_integer(
    "num_samples",
    None,
    "Num of samples to evaluate. Evaluates the entire split if None.",
)

flags.DEFINE_string(
    "webshop_url",
    "http://localhost:3000/",
    "Webshop server URL.",
)

flags.DEFINE_string(
    "cursor_dir",
    "gs://pix2act-data/cursors/",
    "Directory with cursor files.",
)

flags.DEFINE_bool(
    "render_action_history",
    True,
    "Renders action history on the screenshot if true.",
)

flags.DEFINE_integer(
    "num_previous_actions",
    5,
    "Number of previous actions to include in history.",
)

flags.DEFINE_integer(
    "max_action_chars",
    200,
    (
        "Max num of chars which can be rendered on the action section of the"
        " input."
    ),
)

flags.DEFINE_bool(
    "output_html",
    False,
    "Writes predictions to HTML files.",
)


def write_json(filepath, data):
  with tf.io.gfile.GFile(filepath, "w") as fp:
    fp.write(json.dumps(data, indent=2))


def main(unused_argv):
  stub = client_utils.get_stub(FLAGS.server)

  metrics = {
      "num_episodes": 0,
      "num_complete": 0,
      "num_success": 0,
      "task_score": 0.0,
      "success_rate": 0.0,
      "num_all_actions_filtered": 0,
      "rewards": {}
  }

  env_config = webshop_env.get_env_config(FLAGS.cursor_dir)
  chrome_options = webdriver.ChromeOptions()
  driver = env_utils.create_web_driver(chrome_options)
  driver.implicitly_wait(20)
  driver.set_window_size(env_config.width, env_config.height)

  lb, ub = demo_utils.SPLITS[FLAGS.split][0], demo_utils.SPLITS[FLAGS.split][1]
  eval_range = list(range(lb, ub))
  random.shuffle(eval_range)
  eval_range = (
      eval_range[: FLAGS.num_samples] if FLAGS.num_samples else eval_range
  )
  for goal_idx in eval_range:
    split = demo_utils.get_split(goal_idx)
    if split != FLAGS.split:
      continue
    logging.info(
        "Processing %d out of %d, goal: %d",
        metrics["num_episodes"],
        ub - lb,
        goal_idx,
    )
    driver.get(FLAGS.webshop_url + "fixed_%d" % goal_idx)
    instruction_text = demo_utils.get_instruction_text(driver)
    metrics["num_episodes"] += 1

    tfr_filepath = (
        f"{FLAGS.output_dir}/{FLAGS.split}-goal-{goal_idx}.tfr"
    )
    cursor_state = env_utils.CursorState()
    examples = []
    cycle_checker = common_utils.CycleChecker()
    with tf.io.TFRecordWriter(tfr_filepath) as writer:
      step_id = 0
      raw_reward = 0
      # Allow one more "step" for eval.
      while step_id < FLAGS.max_steps + 1:
        if demo_utils.is_episode_done(driver):
          raw_reward = demo_utils.get_reward(driver)
          metrics["num_complete"] += 1
          metrics["task_score"] += raw_reward
          metrics["num_success"] += int(raw_reward == 1.0)
        if demo_utils.is_episode_done(driver) or step_id == FLAGS.max_steps:
          task_score = metrics["task_score"] / metrics["num_episodes"]
          if FLAGS.output_html:
            html_filepath = (
                f"{FLAGS.output_dir}/{FLAGS.split}-goal-{goal_idx}.html"
            )
            viz_utils.write_tf_examples_to_html(
                examples, html_filepath, image_scale=50
            )
          logging.info("Current reward: %f, score: %f", raw_reward, task_score)
          break

        screenshot = env_utils.get_screenshot(
            driver, env_config, cursor_state
        )
        screenshot_bytes = screenshot.tobytes()
        screenshot = render_utils.render_header(
            screenshot, instruction_text, background_color="yellow"
        )
        if FLAGS.render_action_history:
          history = demo_utils.get_action_history(
              examples, FLAGS.num_previous_actions, action_key="action_str"
          )
          screenshot = render_utils.render_action_history(
              screenshot,
              history,
              FLAGS.max_action_chars,
          )
        concatenated_png = render_utils.image_to_png(screenshot)
        beam = client_utils.get_single_beam(stub, concatenated_png)
        predictions = beam.predictions
        example = tf.train.Example()
        if FLAGS.break_cycles:
          predictions = cycle_checker.filter_predictions(
              screenshot_bytes, predictions
          )
          if not predictions:
            logging.info("No more viable predictions.")
            metrics["num_all_actions_filtered"] += 1
            break
        # Take highest scoring action.
        action_str = predictions[-1].output
        cycle_checker.add(screenshot_bytes, action_str)

        # Write step as a TF example for easy visualization with txui.
        tf_utils.add_bytes_feature(example, "screenshot_png", concatenated_png)
        tf_utils.add_text_feature(example, "cursor_state", str(cursor_state))
        tf_utils.add_text_feature(example, "action_str", action_str)
        examples.append(example)
        # One file per example is slow but more robust to server failures
        # which can happen occasionally.
        writer.write(example.SerializeToString())
        logging.info("step_idx: %d, action_str: %s", step_id, action_str)

        # Take the step, and break if unsuccessful.
        try:
          env_utils.process_action(driver, env_config, cursor_state, action_str)
        except Exception as e:
          logging.info("Error executing action: %s", e)
          break
        step_id += 1
      metrics["rewards"][goal_idx] = raw_reward

  metrics["success_rate"] = metrics["num_success"] / metrics["num_episodes"]
  metrics["task_score"] /= metrics["num_episodes"]
  metrics_filepath = f"{FLAGS.output_dir}/{FLAGS.split}.json"
  write_json(metrics_filepath, metrics)


if __name__ == "__main__":
  random.seed(10)
  app.run(main)
