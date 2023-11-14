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

"""Utilities for dealing with MiniWob environment."""

from miniwob import instance as miniwob_instance
from miniwob import reward as miniwob_reward
from pix2act.common import env_utils
from selenium.webdriver.common import by
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support import wait


def get_env_config(cursor_dir):
  return env_utils.EnvConfig(
      cursor_dir=cursor_dir,
      width=160,
      height=210,
      num_x_buckets=32,
      num_y_buckets=32,
      center_buckets=True,
      scrollable_element_fn='document.getElementById("area")',
  )


def normalize_reward(raw_reward):
  """Normalize reward from [-1,1] to [0,1]."""
  return raw_reward / 2.0 + 0.5


def start_episode(instance, base_url, task, seed):
  """Reset instance and start new episode."""
  # Reset task and seed.
  instance.init_seed = repr(seed)
  instance.url = "{}/{}.html".format(base_url, task)
  instance.field_extractor = miniwob_instance.get_field_extractor(task)

  print(f"url: {instance.url}")
  if not hasattr(instance, "driver"):
    instance.create_driver()
  else:
    # Do a hard reset. This method can be called more than once on the same
    # instance.
    instance.force_stop()
    instance.driver.get(instance.url)

    # Ensure page is loaded.
    num_seconds = 5
    wait.WebDriverWait(instance.driver, num_seconds).until(
        expected_conditions.element_to_be_clickable(
            (by.By.ID, instance.SYNC_SCREEN_ID)
        )
    )

    # Reset the seed.
    instance.driver.execute_script(
        "Math.seedrandom({});".format(instance.init_seed)
    )

  # Set long timeout.
  instance.driver.execute_script("core.EPISODE_MAX_TIME=3600000;")
  instance.begin_task()


def get_instance(base_url):
  """Return MiniWoB instance."""

  # Don't use time-decayed rewards.
  reward_processor = miniwob_reward.get_raw_reward

  instance = miniwob_instance.MiniWoBInstance(
      index=0,
      subdomain="",
      seed=None,
      reward_processor=reward_processor,
      base_url=base_url,
      headless=True,
      threading=False,
  )
  return instance


class MiniWobEnv(object):
  """Provides high-level API for MiniWob environment."""

  def __init__(self, miniwob_url, cursor_dir):
    print(f"MiniWob url: {miniwob_url}")
    self.base_url = miniwob_url
    self.env_config = get_env_config(cursor_dir)
    self.cursor_state = None
    self.instance = get_instance(self.base_url)

  def __del__(self):
    self.instance.close()

  def start_episode(self, task, seed):
    self.cursor_state = env_utils.CursorState()
    start_episode(self.instance, self.base_url, task, seed)

  def get_dom(self):
    state = self.instance.get_state()
    return state.dom_info

  def get_utterance(self):
    state = self.instance.get_state()
    return state.utterance

  def get_screenshot_png(self):
    return env_utils.get_screenshot_as_png(
        self.instance.driver, self.env_config, self.cursor_state
    )

  def get_metadata(self):
    return self.instance.get_metadata()

  def is_done(self):
    metadata = self.get_metadata()
    return metadata["done"]

  def get_reward(self):
    metadata = self.instance.get_metadata()
    return metadata["raw_reward"]

  def execute_action(self, action_str):
    env_utils.process_action(
        self.instance.driver, self.env_config, self.cursor_state, action_str
    )
