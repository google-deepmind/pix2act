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

r"""Simple client for testing model server.
"""

from absl import app
from absl import flags
from PIL import Image
from pix2act.common import render_utils
from pix2act.server import client_utils
from tensorflow.io import gfile


FLAGS = flags.FLAGS

flags.DEFINE_string("server", "", "BNS of server.")

flags.DEFINE_string("screenshot", "Path to png.", "")


def main(unused_argv):
  screenshot = Image.open(gfile.GFile(FLAGS.screenshot, "rb"))
  screenshot_png = render_utils.image_to_png(screenshot)
  stub = client_utils.get_stub(FLAGS.server)
  beam = client_utils.get_single_beam(stub, screenshot_png)
  beam_str = client_utils.beam_to_str(beam)
  print(beam_str)


if __name__ == "__main__":
  app.run(main)
