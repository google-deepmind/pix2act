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

"""Run server for Pix2Struct model."""

from concurrent import futures
import threading

from absl import flags
import gin
import grpc
from pix2act.server import inference_utils
from pix2act.server import model_server_pb2
from pix2act.server import model_server_pb2_grpc
from t5x import gin_utils


FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 10000, "Port to listen on.")

flags.DEFINE_string("size", "base", "Model size.")

flags.DEFINE_multi_string("gin_bindings", [], "Individual gin bindings.")

flags.DEFINE_multi_string(
    "gin_search_paths",
    ["pix2act/configs", "pix2struct/configs"],
    "Search paths for gin files.",
)

flags.DEFINE_integer("num_decodes", 1, "Beam size for decoding.")

flags.DEFINE_integer("num_threads", 16, "Number of threads.")


_HOST = "[::]"


def _get_pred_fn():
  """Configure and return prediction function."""
  get_inference_fn_using_gin = gin.configurable(
      inference_utils.get_inference_fn
  )
  gin_utils.parse_gin_flags(
      gin_search_paths=FLAGS.gin_search_paths,
      gin_files=[
          "runs/inference.gin",
          "models/pix2act.gin",
          f"sizes/{FLAGS.size}.gin",
      ],
      gin_bindings=FLAGS.gin_bindings,
  )
  inference_fn = get_inference_fn_using_gin(num_decodes=FLAGS.num_decodes)
  return inference_fn


class ModelServerServicer(model_server_pb2_grpc.ModelServerServicer):
  """A Stubby server."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._predict_fn = _get_pred_fn()
    # Ensure thread-safe access to TPU.
    self._lock = threading.Lock()

  def Predict(
      self,
      request: model_server_pb2.Request,
      unused_context: grpc.ServicerContext,
  ):
    """Generate prediction for request."""
    response = model_server_pb2.Response()
    with self._lock:
      batched_beam_out = self._predict_fn(request.screenshot_png)
      for beam_out in batched_beam_out:
        beam_proto = response.beams.add()
        for output, score in beam_out:
          prediction = beam_proto.predictions.add()
          prediction.output = output
          prediction.score = score
    return response


def _add_port(server: grpc.Server, address: str):
  server.add_insecure_port(address)


def main(unused_argv):
  thread_pool = futures.ThreadPoolExecutor(max_workers=FLAGS.num_threads)
  server = grpc.server(thread_pool)
  servicer = ModelServerServicer()
  model_server_pb2_grpc.add_ModelServerServicer_to_server(servicer, server)
  address = f"{_HOST}:{FLAGS.port}"
  _add_port(server, address)
  server.start()
  server.wait_for_termination()


if __name__ == "__main__":
  gin_utils.run(main)
