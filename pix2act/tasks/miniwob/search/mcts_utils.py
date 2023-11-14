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

"""Utilities related to MCTS."""

import dataclasses
import functools
import math

from absl import logging
from pix2act.common import common_utils
from pix2act.common import env_utils
from pix2act.server import client_utils
from pix2act.server import model_server_pb2_grpc
from pix2act.tasks.miniwob import miniwob_env
from pix2act.tasks.miniwob.search import reward_utils
from pix2act.tasks.miniwob.search import value_fn_utils


@dataclasses.dataclass()
class MctsConfig:
  """Defines static configuration options."""
  exploration_scalar: float
  beam_limit: int
  max_rollout_depth: int
  num_iterations: int
  max_steps: int
  rollout_weight: float


@dataclasses.dataclass()
class SearchContext:
  """Holds important context for running MCTS."""
  mcts_config: MctsConfig
  policy_stub: model_server_pb2_grpc.ModelServerStub
  critic_stub: model_server_pb2_grpc.ModelServerStub
  env: miniwob_env.MiniWobEnv
  seed: int
  task: str


# Cache model server responses.
@functools.cache
def _get_cached_beam(stub, image_bytes):
  return client_utils.get_single_beam(stub, image_bytes)


def _set_env_state(context, action_sequence):
  """Sets the environment state to the given action sequence."""
  # Reset environment.
  context.env.start_episode(task=context.task, seed=context.seed)

  # Take actions in sequence.
  # Unfortunately there is not a clear way to set the environment state directly
  # without taking each action in the sequence.
  for action_str in action_sequence:
    context.env.execute_action(action_str)


def _rollout(node):
  """Perform rollout to approximate value of leaf state."""
  env = node.context.env
  cycle_checker = common_utils.CycleChecker()

  # Critic RPC.
  image_bytes = env.get_screenshot_png()
  critic_beam = _get_cached_beam(node.context.critic_stub, image_bytes)
  value_estimate = value_fn_utils.get_value_prediction(critic_beam)
  logging.info("Value estimate: %.2f", value_estimate)

  rollout_reward = 0.0
  logging.info("Begin rollout...")
  for idx in range(node.context.mcts_config.max_rollout_depth):
    logging.info("Rollout step %s.", idx)
    # Capture information from environment.
    metadata = env.get_metadata()
    image_bytes = env.get_screenshot_png()
    complete = metadata["done"]
    raw_reward = metadata["raw_reward"]

    if complete:
      logging.info("Rollout completed with raw reward: %.2f", raw_reward)
      rollout_reward += miniwob_env.normalize_reward(raw_reward)
      break
    else:
      # Choose next step.
      policy_beam = _get_cached_beam(node.context.policy_stub,
                                     image_bytes)
      predictions = policy_beam.predictions
      predictions = env_utils.filter_predictions(
          node.context.env.env_config, predictions
      )
      predictions = cycle_checker.filter_predictions(image_bytes, predictions)
      action_str = predictions[-1].output
      cycle_checker.add(image_bytes, action_str)
      # Take the action.
      env.execute_action(action_str)
      rollout_reward += reward_utils.STEP_PENALTY

  rollout_reward = max(0.0, rollout_reward)
  rollout_weight = node.context.mcts_config.rollout_weight
  return rollout_weight * rollout_reward + (1 - rollout_weight) * value_estimate


class MctsNode:
  """Represents node in MCTS search tree."""

  def __init__(self, context, parent_node, action):
    self.context = context
    self.action = action
    self.parent_node = parent_node
    if not parent_node:
      # This is the root node.
      self.action_sequence = []
    else:
      self.action_sequence = parent_node.action_sequence + [self.action]

    self._value_network_estimate = None
    self._rollout_estimate = None
    self._value_network_estimate = None
    self._actions = []
    self._action_to_posterior = {}

    _set_env_state(self.context, self.action_sequence)
    self.image_bytes = self.context.env.get_screenshot_png()
    self.metadata = self.context.env.get_metadata()

    if not self.is_complete():
      # Policy RPC.
      policy_beam = _get_cached_beam(self.context.policy_stub,
                                     self.image_bytes)
      truncated_beam = policy_beam.predictions[
          -context.mcts_config.beam_limit :
      ]
      for prediction in truncated_beam:
        if not env_utils.is_valid(context.env.env_config, prediction.output):
          continue
        self._actions.append(prediction.output)
        posterior = math.exp(prediction.score)
        self._action_to_posterior[prediction.output] = posterior

    self._value_estimate = _rollout(self)

    self._action_to_count = {k: 0 for k in self._actions}
    self._action_to_child = {k: None for k in self._actions}
    self._action_to_reward = {k: 0.0 for k in self._actions}

  def image(self):
    return self.image_bytes

  def action_str(self):
    return self.action

  def total_count(self):
    return sum(self._action_to_count.values())

  def get_max_child(self):
    """Return highest scoring child after simulations."""
    logging.info("Selecting next move.")
    max_value = None
    max_action = None
    for action in self._actions:
      num_visits = self._action_to_count[action]
      if num_visits:
        avg_reward = self._action_to_reward[action] / num_visits
      else:
        avg_reward = 0.0
      # Rank by num visits and use average return to break ties.
      value = (num_visits, avg_reward)
      logging.info("%s: (%s, %.2f)", action, num_visits, avg_reward)
      if not max_value or value > max_value:
        max_value = value
        max_action = action
    logging.info("Max action: %s", max_action)
    return self._action_to_child[max_action]

  def get_value(self):
    return self._value_estimate

  def is_complete(self):
    return self.metadata["done"]

  def raw_reward(self):
    return self.metadata["raw_reward"]

  def has_children(self):
    return bool(self._actions)

  def get_child(self, action):
    return self._action_to_child[action]

  def add_child(self, action):
    child = MctsNode(self.context, self, action)
    self._action_to_child[action] = child
    return child

  def _compute_uct(self, action):
    """Computes Upper Confidence function for state-action."""
    total_count = self.total_count()
    action_count = self._action_to_count[action]
    posterior = self._action_to_posterior[action]
    reward = self._action_to_reward[action]
    # We compute the average reward from selecting the action.
    if action_count:
      action_value = reward / action_count
    else:
      # Default assumption prior to exploration.
      action_value = self.get_value()
    # Bonus term encourages exploration of actions with fewer visits, scaled
    # by model posterior.
    bonus = (
        self.context.mcts_config.exploration_scalar
        * posterior
        * math.sqrt(total_count + 1)
        / (action_count + 1)
    )
    uct = action_value + bonus
    logging.info(
        "%s uct (%s): %.2f + %.3f = %.2f", action,
        action_count, action_value, bonus, uct
    )
    return uct

  def select_action(self):
    """Choose action that maximizes UCT."""
    max_uct = None
    max_action = None
    for action in self._actions:
      uct = self._compute_uct(action)
      if not max_uct or uct > max_uct:
        max_uct = uct
        max_action = action
    return max_action

  def backprop_value(self):
    self.parent_node.backprop(
        self.action, self.get_value() + reward_utils.STEP_PENALTY
    )

  def backprop(self, action, reward):
    logging.info("Backprop `%.2f` to `%s`", reward, action)
    self._action_to_count[action] += 1
    self._action_to_reward[action] += reward
    if self.parent_node:
      self.parent_node.backprop(self.action, reward + reward_utils.STEP_PENALTY)

  def __str__(self):
    return "%s (%.2f)" % (self.action_sequence, self.get_value())


def _run_simulations(context, root_node):
  """Implement MCTS simulations."""
  while root_node.total_count() < context.mcts_config.num_iterations:
    logging.info("Simulations: %s.", root_node.total_count())
    # Selection.
    current_node = root_node
    while True:
      logging.info("Selected node: %s", current_node)
      action = current_node.select_action()
      next_node = current_node.get_child(action)
      if not next_node:
        # Expand this node then backprop.
        next_node = current_node.add_child(action)
        logging.info("Expanded node: %s", next_node)
        next_node.backprop_value()
        break
      elif not next_node.has_children():
        # Reached a terminal state, just backprop.
        logging.info("Reached terminal node: %s", next_node)
        next_node.backprop_value()
        break
      else:
        # Keep selecting next node.
        current_node = next_node
  logging.info("Max iterations reached.")
  return None


def run_mcts(context: SearchContext) -> list[MctsNode]:
  """Iteratively select actions according to MCTS simulations."""
  current_node = MctsNode(context, None, None)

  step_nodes = []
  for step in range(context.mcts_config.max_steps):
    logging.info("Step %s.", step)
    logging.info("Current node: %s", str(current_node.action_sequence))
    step_nodes.append(current_node)

    if current_node.is_complete():
      logging.info("Completed episode.")
      break
    _run_simulations(context, current_node)
    # Select highest scoring node according to MCTS.
    current_node = current_node.get_max_child()

  return step_nodes
