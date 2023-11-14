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

"""Utilities related to surrogate reward used by value function approximator."""

from pix2act.tasks.miniwob import miniwob_env


STEP_PENALTY = -1.0 / 30
VALUE_FN_SCALAR = 30
_NORMALIZED_REWARD_THRESHOLD = 0.9


def compute_surrogate_reward(raw_reward, steps_to_go=0):
  normalized_reward = miniwob_env.normalize_reward(raw_reward)
  step_penality = STEP_PENALTY * steps_to_go
  if normalized_reward < _NORMALIZED_REWARD_THRESHOLD:
    normalized_reward = 0.0
  surrogate_reward = normalized_reward + step_penality
  return surrogate_reward


def surrogate_reward_to_value_fn_target(surrogate_reward):
  """Map surrogate reward to value fn range."""
  value_fn_output = round(surrogate_reward * VALUE_FN_SCALAR)
  return value_fn_output


def value_fn_output_to_surrogate_reward(value_fn_output):
  surrogate_reward = value_fn_output / VALUE_FN_SCALAR
  return surrogate_reward
