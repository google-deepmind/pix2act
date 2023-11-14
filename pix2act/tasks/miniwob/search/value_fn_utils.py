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

"""Utilities for value function approximator."""

import math

from pix2act.tasks.miniwob.search import reward_utils


def _get_raw_value_prediction(beam, marginal, max_steps, kbest):
  """Optionally use beam search approximation to distribution mean."""
  allowable_outputs = set(str(i) for i in range(max_steps))
  if not marginal:
    # Just return highest scoring prediction.
    output = beam.predictions[-1].output
    if output not in allowable_outputs:
      raise ValueError("Unexpected prediction: `%s`" % output)
    return float(output)

  # Otherwise compute marginal prediction.
  outputs = []
  weights = []
  for prediction in beam.predictions[-kbest:]:
    if prediction.output not in allowable_outputs:
      continue
    outputs.append(float(prediction.output))
    weights.append(math.exp(prediction.score))
  sum_weights = sum(weights)
  normalized_weights = [weight / sum_weights for weight in weights]
  marginal_pred = 0.0
  for output, normalized_weight in zip(outputs, normalized_weights):
    marginal_pred += output * normalized_weight
  return marginal_pred


def get_value_prediction(beam, marginal=True, max_steps=30, kbest=3):
  raw_value_prediction = _get_raw_value_prediction(
      beam, marginal, max_steps, kbest
  )
  return reward_utils.value_fn_output_to_surrogate_reward(raw_value_prediction)
