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

"""Utilities for dealing with web environment."""

import dataclasses

from pix2act.common import render_utils
from selenium import webdriver
from selenium.webdriver.common import action_chains
from selenium.webdriver.common import keys


@dataclasses.dataclass()
class EnvConfig:
  """Web environment configuration."""
  # Directory with cursor graphics.
  cursor_dir: str
  # Width of viewable area.
  width: int
  # Height of viewable area.
  height: int
  # Number of horizontal coordinate buckets.
  num_x_buckets: int
  # Number of vertical coordinate buckets.
  num_y_buckets: int
  # Whether clicks refer to center or upper left of coordinate bucket.
  center_buckets: bool
  # Function to return element that scroll actions should be applied to.
  scrollable_element_fn: str


@dataclasses.dataclass()
class CursorState:
  """Tracks state of cursor."""
  # Whether mouse pointer is currently "down".
  ptr_down: bool = False
  # Cursor x position.
  ptr_x: int = 10
  # Cursor y position.
  ptr_y: int = 10
  # Cursor style.
  cursor: str = "pointer"


def rel_x_y_to_x_y(env_config, x_rel, y_rel):
  """Map coordinate buckets to pixel coordinates."""
  add = 0.5 if env_config.center_buckets else 0.0
  x = (float(x_rel) + add) / env_config.num_x_buckets * env_config.width
  y = (float(y_rel) + add) / env_config.num_y_buckets * env_config.height
  return round(x), round(y)


def is_float(element: str) -> bool:
  """Returns whether `element` can be parsed as a float."""
  try:
    float(element)
    return True
  except ValueError:
    return False


def is_valid_coordinate(env_config: EnvConfig, x_str: str, y_str: str) -> bool:
  """Returns whether the given strings are valid coordinate buckets."""
  if not is_float(x_str) or not is_float(y_str):
    return False
  x = float(x_str)
  y = float(y_str)

  return (
      x >= 0
      and y >= 0
      and x < env_config.num_x_buckets
      and y < env_config.num_y_buckets
  )


def is_valid(env_config, action_str):
  """Return True if `action_str` is valid."""
  if action_str.startswith("double_click"):
    split = action_str.split()
    return len(split) == 3 and is_valid_coordinate(
        env_config, split[1], split[2]
    )
  elif action_str.startswith("click"):
    split = action_str.split()
    return len(split) == 3 and is_valid_coordinate(
        env_config, split[1], split[2]
    )
  elif action_str.startswith("begin_drag"):
    split = action_str.split()
    return len(split) == 3 and is_valid_coordinate(
        env_config, split[1], split[2]
    )
  elif action_str.startswith("end_drag"):
    split = action_str.split()
    return len(split) == 3 and is_valid_coordinate(
        env_config, split[1], split[2]
    )
  elif action_str.startswith("scroll"):
    split = action_str.split()
    return len(split) == 2 and is_float(split[1])
  elif action_str.startswith("keys"):
    return True
  elif action_str.startswith("key"):
    return True
  else:
    return False


def filter_predictions(env_config, predictions):
  filtered_predictions = []
  for prediction in predictions:
    if is_valid(env_config, prediction.output):
      filtered_predictions.append(prediction)
  return filtered_predictions


def process_action(
    driver,
    env_config,
    cursor_state,
    action_str,
):
  """Executes action based on string representation of action."""
  if action_str.startswith("double_click"):
    _, x_rel, y_rel = action_str.split()
    x, y = rel_x_y_to_x_y(env_config, x_rel, y_rel)
    _ptr_move(driver, cursor_state, x, y)
    _double_click(driver)
  elif action_str.startswith("click"):
    _, x_rel, y_rel = action_str.split()
    x, y = rel_x_y_to_x_y(env_config, x_rel, y_rel)
    _click(driver, cursor_state, x, y)
  elif action_str.startswith("begin_drag"):
    _, x_rel, y_rel = action_str.split()
    x, y = rel_x_y_to_x_y(env_config, x_rel, y_rel)
    _ptr_move(driver, cursor_state, x, y)
    _ptr_down(driver, cursor_state)
  elif action_str.startswith("end_drag"):
    _, x_rel, y_rel = action_str.split()
    x, y = rel_x_y_to_x_y(env_config, x_rel, y_rel)
    _ptr_move(driver, cursor_state, x, y)
    _ptr_up(driver, cursor_state)
  elif action_str.startswith("scroll"):
    _, delta_y = action_str.split()
    _scroll(driver, int(delta_y), env_config.scrollable_element_fn)
  elif action_str.startswith("keys"):
    splits = action_str.split()
    _key_press(driver, " ".join(splits[1:]))
  elif action_str.startswith("key"):
    splits = action_str.split()
    if len(splits) == 2:
      _, key = splits
      _key_press(driver, key)
    elif len(splits) == 3:
      _, key_hold, key = splits
      _key_press(driver, key, key_hold=key_hold)
    else:
      raise ValueError
  cursor_state.cursor = get_cursor_type(driver, cursor_state)


def _key_press(driver, key, key_hold=None):
  """Execute key press."""
  # Some keys require special handling.
  key_map = {
      "Backspace": keys.Keys.BACKSPACE,
      "ArrowRight": keys.Keys.ARROW_RIGHT,
      "ArrowLeft": keys.Keys.ARROW_LEFT,
      "ArrowDown": keys.Keys.ARROW_DOWN,
      "ArrowUp": keys.Keys.ARROW_UP,
      "Tab": keys.Keys.TAB,
      "Space": keys.Keys.SPACE,
      "Enter": keys.Keys.ENTER,
  }
  key_hold_map = {
      "ctrl": keys.Keys.CONTROL,
      "shift": keys.Keys.SHIFT,
  }
  key = key_map.get(key, key)
  if key_hold:
    if key_hold not in key_hold_map:
      raise ValueError
    key_hold = key_hold_map[key_hold]

  chain = action_chains.ActionChains(driver)
  if key_hold:
    chain.key_down(key_hold).send_keys(key).key_up(key_hold).perform()
  else:
    chain.send_keys(key).perform()


def _scroll(driver, delta_y, get_scrollable_element_fn):
  # Note that `scroll_by_amount` is supported in Selenium version 4.8 and
  # later, but not by the version of Selenium used for this project.
  # Regardless, we implement scrolling by executing javascript.
  command = "%s.scrollBy(0,%d);" % (get_scrollable_element_fn, delta_y)
  driver.execute_script(command)


def _ptr_move(driver, cursor_state, x, y):
  chain = action_chains.ActionChains(driver)
  chain.w3c_actions.pointer_action.move_to_location(x, y)
  chain.w3c_actions.perform()
  cursor_state.ptr_x = x
  cursor_state.ptr_y = y


def _ptr_down(driver, cursor_state):
  chain = action_chains.ActionChains(driver)
  chain.w3c_actions.pointer_action.pointer_down()
  chain.w3c_actions.perform()
  cursor_state.ptr_down = True


def _ptr_up(driver, cursor_state):
  chain = action_chains.ActionChains(driver)
  chain.w3c_actions.pointer_action.pointer_up()
  chain.w3c_actions.perform()
  cursor_state.ptr_down = False


def _double_click(driver):
  chain = action_chains.ActionChains(driver)
  chain.w3c_actions.pointer_action.double_click()
  chain.w3c_actions.perform()


def _click(driver, cursor_state, x, y):
  chain = action_chains.ActionChains(driver)
  chain.w3c_actions.pointer_action.move_to_location(x, y)
  chain.w3c_actions.pointer_action.click()
  cursor_state.ptr_x = x
  cursor_state.ptr_y = y
  chain.w3c_actions.perform()


def create_web_driver(chrome_options):
  chrome_options.add_argument("disable-gpu")
  chrome_options.add_argument("headless")
  print("chrome_options: %s" % chrome_options.arguments)
  return webdriver.Chrome(options=chrome_options)


def get_screenshot(driver, env_config, cursor_state):
  """Returns screenshot in the form of Image.Image."""
  screenshot = render_utils.png_to_image(driver.get_screenshot_as_png())
  screenshot = render_utils.crop(
      screenshot, env_config.width, env_config.height
  )
  screenshot = render_utils.add_cursor(
      env_config.cursor_dir, screenshot, cursor_state
  )
  screenshot = render_utils.augment_screenshot(
      screenshot, cursor_state.ptr_down
  )
  return screenshot


def get_screenshot_as_png(driver, env_config, cursor_state):
  """Returns screenshot in the form of png."""
  screenshot = get_screenshot(driver, env_config, cursor_state)
  return render_utils.image_to_png(screenshot)


def get_cursor_type(driver, cursor_state):
  """Returns cursor type for focused element, e.g. "pointer", "default", etc."""
  command = (
      "return"
      f" document.elementFromPoint({cursor_state.ptr_x},{cursor_state.ptr_y});"
  )
  element = driver.execute_script(command)
  cursor_type = element.value_of_css_property("cursor")
  return cursor_type
