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

"""Utilities related to WebShop environment."""

from pix2act.common import env_utils


WIDTH = 800
HEIGHT = 600
NUM_X_BUCKETS = 100
NUM_Y_BUCKETS = 100
CENTER_BUCKETS = False
SCROLLABLE_ELEMENT_FN = "window"


def get_env_config(cursor_dir):
  return env_utils.EnvConfig(
      cursor_dir=cursor_dir,
      width=WIDTH,
      height=HEIGHT,
      num_x_buckets=NUM_X_BUCKETS,
      num_y_buckets=NUM_Y_BUCKETS,
      center_buckets=CENTER_BUCKETS,
      scrollable_element_fn=SCROLLABLE_ELEMENT_FN,
  )
