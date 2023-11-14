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

r"""Converts demonstrations to tf examples for training, validation, and test.

# pylint:disable=long-too-long
This requires that the Webshop server is running locally. See the official repo
for setup instructions: https://github.com/princeton-nlp/WebShop

Follows split and preprocessing here from the get_data method here:
https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/train_choice_il.py

# pylint:enable=long-too-long
"""
import json
import os
import typing
from typing import Any, Dict, List

from absl import app
from absl import flags
from pix2act.common import env_utils
from pix2act.common import render_utils
from pix2act.common import tf_utils
from pix2act.tasks.webshop import demo_utils
from pix2act.tasks.webshop import webshop_env
from selenium import webdriver
from selenium.common import exceptions
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "webshop_url",
    "http://localhost:3000/",
    "Webshop server URL.",
)

flags.DEFINE_string(
    "demo_file",
    "",
    "File containing high-level demonstrations.",
)

flags.DEFINE_string(
    "human_goals_file",
    "",
    "Human goals file which dictates train/dev/test split.",
)

flags.DEFINE_string(
    "processed_dir",
    "",
    "Processed dir name.",
)

flags.DEFINE_float(
    "reward_threshold",
    0.1,
    "Demonstrations below this threshold will be discarded.",
)

flags.DEFINE_bool(
    "do_word_input_search",
    True,
    "Use word level input for search.",
)

flags.DEFINE_bool(
    "skip_test",
    True,
    "Skips test split if true.",
)

flags.DEFINE_string(
    "cursor_dir",
    "gs://pix2act-data/cursors/",
    "Directory with cursor files.",
)

flags.DEFINE_bool(
    "render_action_history",
    False,
    "Renders action history on the screenshot if true.",
)

flags.DEFINE_integer(
    "num_prepend_actions",
    5,
    "Prepends these many previous actions to parse before current actions.",
)

flags.DEFINE_integer(
    "max_action_chars",
    200,
    (
        "Max num of chars which can be rendered on the action section of the"
        " input."
    ),
)


def process_data(driver):
  """Process and split data according to the official WebShop repo."""
  env_config = webshop_env.get_env_config(FLAGS.cursor_dir)
  demos = demo_utils.read_demos_file(FLAGS.demo_file)
  human_goals = demo_utils.read_goals_file(FLAGS.human_goals_file)
  split_info = {}
  for split in demo_utils.SPLITS.keys():
    split_info[split] = {"num_processed": 0, "rewards": {}}
  for demo_idx, demo in enumerate(demos):
    demo_examples = []
    _, goal_idx = demo_utils.process_goal(demo["states"][0], human_goals)
    split = demo_utils.get_split(goal_idx)
    if FLAGS.skip_test and split == "test":
      continue

    print("Processing %d out of %d" % (demo_idx, len(demos)))
    driver.get(FLAGS.webshop_url + "fixed_%d" % goal_idx)
    instruction_text = demo_utils.get_instruction_text(driver)
    cursor_state = env_utils.CursorState()
    for i, (demo_action, demo_action_translate) in enumerate(
        zip(demo["actions"], demo["actions_translate"])
    ):
      for low_level_action, _ in demo_utils.convert_action(
          driver, demo_action, demo_action_translate, FLAGS.do_word_input_search
      ):
        parse = low_level_action
        history = demo_utils.get_action_history(
            demo_examples,
            FLAGS.num_prepend_actions,
        )
        current_frame = env_utils.get_screenshot(
            driver, env_config, cursor_state
        )
        current_frame = render_utils.render_header(
            current_frame, instruction_text, "yellow"
        )
        if FLAGS.render_action_history:
          current_frame = render_utils.render_action_history(
              current_frame,
              history,
              FLAGS.max_action_chars,
          )
        try:
          env_utils.process_action(
              driver,
              env_config,
              cursor_state,
              low_level_action,
          )
        except exceptions.WebDriverException as e:
          print(e.msg)
          print(
              "Error in demo %d, high-level action %d, %s and low-level"
              " action %s" % (demo_idx, i, demo_action, low_level_action)
          )
        example = tf.train.Example()
        tf_utils.add_text_feature(
            example, "id", "%d_%d_%d" % (demo_idx, goal_idx, i)
        )
        tf_utils.add_text_feature(example, "parse", parse)
        tf_utils.add_bytes_feature(
            example, "image", render_utils.image_to_png(current_frame)
        )
        demo_examples.append(example)
    split_info[split]["num_processed"] += 1
    reward = demo_utils.get_reward(driver)
    write_processed_info({"reward": reward, "goal": goal_idx}, split, demo_idx)
    print("Demo: %d, reward: %f" % (demo_idx, reward))
    if reward >= FLAGS.reward_threshold:
      write_processed_data(demo_examples, split, demo_idx)
    split_info[split]["rewards"][demo_idx] = reward
  for split in demo_utils.SPLITS:
    write_processed_info(split_info[split], split)
  return


def write_processed_info(
    processed_info: Dict[str, Any],
    split: str,
    index: typing.Optional[int] = None,
):
  file_prefix = split if index is None else split + "-demo-%d" % index
  with tf.io.gfile.GFile(
      os.path.join(FLAGS.processed_dir, file_prefix + ".json"),
      "w",
  ) as f:
    f.write(json.dumps(processed_info, indent=2, sort_keys=True))


def write_processed_data(
    examples: List[tf.train.Example],
    split: str,
    index: typing.Optional[int] = None,
):
  """Write processed data to file."""
  file_prefix = split if index is None else split + "-demo-%d" % index
  with tf.io.TFRecordWriter(
      os.path.join(FLAGS.processed_dir, file_prefix + ".tfr")
  ) as writer:
    for example in examples:
      writer.write(example.SerializeToString())


def main(_):
  chrome_options = webdriver.ChromeOptions()
  driver = env_utils.create_web_driver(chrome_options)
  driver.implicitly_wait(20)
  driver.set_window_size(webshop_env.WIDTH, webshop_env.HEIGHT)
  process_data(driver)
  driver.quit()


if __name__ == "__main__":
  app.run(main)
