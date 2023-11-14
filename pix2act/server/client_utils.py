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

"""Utilities for model server clients."""

import math
import grpc
from pix2act.common import render_utils
from pix2act.server import model_server_pb2
from pix2act.server import model_server_pb2_grpc

# Default deadline (timeout) in seconds.
DEFAULT_DEADLINE = 180.0


def _get_channel(address):
  return grpc.insecure_channel(address)


def get_stub(address: str) -> model_server_pb2_grpc.ModelServerStub:
  channel = _get_channel(address)
  grpc.channel_ready_future(channel).result()
  return model_server_pb2_grpc.ModelServerStub(channel)


def get_response(stub, screenshot_png_batch, deadline=DEFAULT_DEADLINE):
  """Get response from model server.

  Args:
    stub: model_server_pb2.ModelServerStub
    screenshot_png_batch: A sequence of bytes representing screenshot pngs.
    deadline: Int deadline in seconds.

  Returns:
    model_server_pb2.Response
  Raises:
    grpc.RpcError if RPC errors.
  """
  request = model_server_pb2.Request(screenshot_png=screenshot_png_batch)
  return stub.Predict(request, timeout=deadline)


def get_pad_input():
  # Empty image is used for padding.
  return render_utils.image_to_png(
      render_utils.create_empty_image_of_size((100, 100))
  )


def get_single_beam(stub, screenshot_png, deadline=DEFAULT_DEADLINE):
  response = get_response(stub, [screenshot_png], deadline)
  if len(response.beams) != 1:
    raise ValueError("Unexpected number of outputs.")
  return response.beams[0]


def beam_to_str(beam):
  """Returns a string representation of the beam predictions for debugging."""
  return "\n".join(
      [
          "%s: %.2f" % (prediction.output, math.exp(prediction.score))
          for prediction in beam.predictions
      ]
  )
