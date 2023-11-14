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

"""Utilities for converting high-level demo actions to low-level Selenium ones."""
import json
from typing import Iterable, Tuple
from pix2act.common import tf_utils
from pix2act.tasks.webshop import webshop_env
import tensorflow as tf


ACTION_BUTTONS = frozenset([
    "back to search",
    "next >",
    "< prev",
    "description",
    "features",
    "reviews",
    "attributes",
    "buy now",
])
_EPISODE_END_TEXT = "Thank you for shopping with us!"
SPLITS = {"train": (1500, 12087), "val": (500, 1500), "test": (0, 500)}


def maybe_prepend_instruction(
    instruction_text: str, prepend_instruction: int
) -> str:
  return "%s <s> " % instruction_text if prepend_instruction else ""


def format_action(action: str) -> str:
  return action[:50]


def get_action_history(
    prev_examples,
    num_previous_actions,
    action_key="parse") -> str:
  """Get action history as a string."""
  history = ""
  prev_examples = prev_examples[
      max(0, len(prev_examples) - num_previous_actions) :
  ]
  prior_actions = [
      tf_utils.get_text_feature(prev_example, action_key).split(" <s> ")[-1]
      for prev_example in prev_examples
  ]
  prior_actions = [format_action(action) for action in prior_actions]
  # Placeholder actions for when there is little or no action history.
  for _ in range(num_previous_actions - len(prior_actions)):
    history += "none <s> "
  history += " <s> ".join(prior_actions)
  history += " <s> " if prior_actions else ""
  return history


def process_goal(state, human_goals):
  """Extracts goal from state (following the official repo)."""
  state = state.lower().replace('"', "").replace("'", "")
  state = state.replace("amazon shopping game\ninstruction:", "").replace(
      "webshop\ninstruction:", ""
  )
  goal = state.replace("\n[button] search [button_]", "").strip()
  if ", and price lower than" in goal:
    goal = goal.split(", and price lower than")[0]
  assert goal in human_goals, goal
  goal_idx = human_goals.index(goal)
  return goal, goal_idx


def get_split(goal_idx):
  for split, (i, j) in SPLITS.items():
    if i <= goal_idx < j:
      return split


def read_goals_file(human_goals_file: str):
  with tf.io.gfile.GFile(human_goals_file, "r") as f:
    goals = json.loads(f.read())
  # Make sure the number of goals matches split information.
  assert len(goals) == SPLITS["train"][1], (len(goals), SPLITS["train"][1])
  return goals


def read_demos_file(demo_file: str):
  demos = []
  with tf.io.gfile.GFile(demo_file, "r") as f:
    for line in f:
      demos.append(json.loads(line))
  return demos


def get_reward(driver) -> float:
  element = driver.find_element(by="id", value="reward")
  element = element.find_element(by="tag name", value="pre")
  return float(element.text)


def is_episode_done(driver):
  return _EPISODE_END_TEXT in driver.page_source


def get_instruction_text(driver) -> str:
  element = driver.find_element(by="id", value="instruction-text")
  return element.text[len("Instruction:"):].strip()


def x_y_to_rel_x_y(x: int, y: int) -> Tuple[int, int]:
  x_rel = x * webshop_env.NUM_X_BUCKETS / webshop_env.WIDTH
  y_rel = y * webshop_env.NUM_Y_BUCKETS / webshop_env.HEIGHT
  return int(x_rel), int(y_rel)


def get_click_coordinates(driver, element) -> Tuple[int, int]:
  x, y = element.location["x"], element.location["y"]
  x_offset = driver.execute_script("return window.pageXOffset;")
  y_offset = driver.execute_script("return window.pageYOffset;")
  height, width = element.size["height"], element.size["width"]
  return x_y_to_rel_x_y(x - x_offset + width / 2, y - y_offset + height / 2)


def _normalize_str(x):
  x = " ".join(x.split())
  return x.lower()


def get_element(driver, arg, arg_translated):
  """Get the HTML element referred to in the arg."""
  arg = _normalize_str(arg)
  arg_translated = _normalize_str(arg_translated)
  if arg in ACTION_BUTTONS:
    elements = driver.find_elements(by="tag name", value="button")
    element = next((e for e in elements if _normalize_str(e.text) == arg), None)
    assert element is not None, (arg, arg_translated)
    return element
  elif arg_translated.startswith("item - "):
    return driver.find_element(by="link text", value=arg.upper())
  else:
    elements = driver.find_elements(by="tag name", value="label")
    element = next((e for e in elements if _normalize_str(e.text) == arg), None)
    assert element is not None, (arg, arg_translated)
    return element


def is_coordinate_not_in_full_view(click_y: int) -> bool:
  return coordinate_needs_up_scroll(click_y) or coordinate_needs_down_scroll(
      click_y
  )


def coordinate_needs_down_scroll(click_y: int) -> bool:
  # Allow some margin below the element.
  return click_y > webshop_env.NUM_Y_BUCKETS - webshop_env.NUM_Y_BUCKETS / 10


def coordinate_needs_up_scroll(click_y: int) -> bool:
  return click_y < 0


def is_scrollable(driver, scroll_y: int) -> bool:
  scrolled_height = driver.execute_script(
      "return window.pageYOffset + window.innerHeight"
  )
  if scroll_y < 0:
    return scrolled_height != webshop_env.HEIGHT
  else:
    page_height = driver.execute_script("return document.body.scrollHeight")
    return page_height - 1 > scrolled_height


def convert_action(
    driver, action, action_translated, word_input_to_search
) -> Iterable[Tuple[str, str]]:
  """Convert high-level action to an iterable over low-level actions."""
  if action.startswith("click"):
    arg = action[len("click[") : -1]
    arg_translated = action_translated[len("click[") : -1]
    element = get_element(driver, arg, arg_translated)
    click_x, click_y = get_click_coordinates(driver, element)

    # We decide scroll direction only at the beginning for simplicity.
    scroll_y = (
        int(webshop_env.HEIGHT / 2)
        if coordinate_needs_down_scroll(click_y)
        else int(-webshop_env.HEIGHT / 2)
    )

    while is_coordinate_not_in_full_view(click_y) and is_scrollable(
        driver, scroll_y
    ):
      yield "scroll %d" % scroll_y, "scroll"
      click_x, click_y = get_click_coordinates(driver, element)
    yield "click %s %s" % (click_x, click_y), "click %s" % element.text
  elif action.startswith("search"):
    query = action[len("search[") : -1]
    element = driver.find_element(by="id", value="search_input")
    # Click on the centre of the search box.
    click_x, click_y = get_click_coordinates(driver, element)
    yield "click %s %s" % (click_x, click_y), "click %s" % element.text

    # Send query to the search box char-by-char.
    if word_input_to_search:
      yield "keys %s" % query.strip(), "keys"
    else:
      qtokens = query.split()
      for token in qtokens:
        for key in token:
          yield "key %s" % key, "key"
        yield "key Space", "key"
    # Submit.
    yield "key Enter", "key"
  else:
    raise ValueError("Unexpected action: `%s`" % action)
