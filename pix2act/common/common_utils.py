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

"""Common utilities."""

import collections


class CycleChecker:
  """Caches previous state to avoid cycles."""

  def __init__(self):
    self._state_to_actions = collections.defaultdict(set)

  def add(self, current_frame_bytes, action_str):
    """Return True if frame is repeated."""
    self._state_to_actions[current_frame_bytes].add(action_str)

  def is_seen(self, current_frame_bytes):
    return current_frame_bytes in self._state_to_actions

  def filter_predictions(self, current_frame_bytes, predictions):
    """Removes actions previously taken at same frame."""
    previous_actions = self._state_to_actions[current_frame_bytes]
    filtered_predictions = []
    for pred in predictions:
      if pred.output in previous_actions:
        print("Removing previous action: %s" % pred.output)
      else:
        filtered_predictions.append(pred)
    return filtered_predictions
