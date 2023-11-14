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

"""Tests for mcts_utils.

This test provides only a simple check of basic functionality but could be
extended.
"""

from pix2act.common import env_utils
from pix2act.server import model_server_pb2
from pix2act.tasks.miniwob.search import mcts_utils
import tensorflow as tf


class MockEnvironment:
  """Mock verion of MiniWobEnv."""

  def __init__(self):
    self.actions = []
    self.env_config = env_utils.EnvConfig(
        cursor_dir="",
        width=160,
        height=210,
        num_x_buckets=32,
        num_y_buckets=32,
        center_buckets=True,
        scrollable_element_fn="",
    )
    # Used to generate unique states.
    self.counter = 0

  def start_episode(self, task, seed):
    del task, seed
    self.actions = []
    self.counter = 0

  def execute_action(self, action_str):
    self.actions.append(action_str)

  def get_screenshot_png(self):
    mock_screenshot = bytes(self.counter)
    self.counter += 1
    return mock_screenshot

  def get_metadata(self):
    if self.actions == ["click 1 1", "click 1 2"]:
      return {
          "done": True,
          "raw_reward": 1.0,
      }
    else:
      return {
          "done": False,
          "raw_reward": -1.0,
      }


class MockPolicyStub:
  """Mock of `model_server_pb2_grpc.ModelServerStub` for policy model."""

  def Predict(  # pylint:disable=invalid-name
      self, unused_request, *unused_args, **unused_kwargs
  ):
    """Generate prediction for request."""
    response = model_server_pb2.Response()
    beam_proto = response.beams.add()
    prediction = beam_proto.predictions.add()
    prediction.output = "click 1 1"
    prediction.score = -1.0
    prediction = beam_proto.predictions.add()
    prediction.output = "click 1 2"
    prediction.score = -2.0
    return response


class MockCriticStub:
  """Mock of `model_server_pb2_grpc.ModelServerStub` for critic model."""

  def Predict(  # pylint:disable=invalid-name
      self, unused_request, *unused_args, **unused_kwargs
  ):
    """Generate prediction for request."""
    response = model_server_pb2.Response()
    beam_proto = response.beams.add()
    prediction = beam_proto.predictions.add()
    prediction.output = "1"
    prediction.score = -1.0
    prediction = beam_proto.predictions.add()
    prediction.output = "2"
    prediction.score = -2.0
    return response


class MctsUtilsTest(tf.test.TestCase):

  def test_run_mcst(self):
    mcts_config = mcts_utils.MctsConfig(
        exploration_scalar=0.1,
        beam_limit=2,
        max_rollout_depth=4,
        num_iterations=4,
        max_steps=4,
        rollout_weight=0.5,
    )
    # We set `seed` and `task` to None because they are not needed when using
    # `MockEnvironment`.
    context = mcts_utils.SearchContext(
        mcts_config=mcts_config,
        policy_stub=MockPolicyStub(),
        critic_stub=MockCriticStub(),
        env=MockEnvironment(),
        seed=None,
        task=None,
    )
    nodes = mcts_utils.run_mcts(context)
    # MCTS should have found the trajectory leading to the completed state.
    self.assertLen(nodes, 3)
    # The first node is the root state and does not have an action string.
    self.assertIsNone(nodes[0].action_str())
    self.assertEqual(nodes[1].action_str(), "click 1 1")
    self.assertEqual(nodes[2].action_str(), "click 1 2")


if __name__ == "__main__":
  tf.test.main()
