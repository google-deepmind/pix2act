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

"""Utilities functions related to rendering screenshots."""

import functools
import io
import os
from typing import Tuple

from PIL import Image
from PIL import ImageDraw
from pix2struct.preprocessing import preprocessing_utils
from tensorflow.io import gfile


@functools.cache
def get_cursor(cursor_dir: str, filename: str) -> Image.Image:
  cursor_path = os.path.join(cursor_dir, filename)
  return Image.open(gfile.GFile(cursor_path, "rb"))


def image_to_png(image: Image.Image) -> bytes:
  bytes_buffer = io.BytesIO()
  image.save(bytes_buffer, "png")
  return bytes_buffer.getvalue()


def png_to_image(image_png: bytes) -> Image.Image:
  return Image.open(io.BytesIO(image_png))


def create_empty_image_of_size(size: Tuple[int, int]) -> Image.Image:
  """Create an empty image with the given size."""
  return Image.new("RGB", size)


def crop(screenshot: Image.Image, width: int, height: int) -> Image.Image:
  screenshot = screenshot.crop((0, 0, width, height))
  return screenshot


def add_cursor(
    cursor_dir: str, screenshot: Image.Image, cursor_state
) -> Image.Image:
  """Renders a cursor on top of the screenshot.

  Arguments:
    cursor_dir: Directory for cursors.
    screenshot: Should be Image.Image corresponding to screenshot.
    cursor_state: CursorState object.

  Returns:
    Bytes corresponding to screenshot png with cursor.
  """
  # Specify small offset for cursor graphic that attempts to approximate
  # how OS renders cursors.
  # Currently we only support the default cursor and the "crosshair" cursor
  # which is commonly used in MiniWob for elements such as the color picker.
  if cursor_state.cursor == "crosshair":
    file = "cross.png"
    x_offset = 10
    y_offset = 9
  else:
    file = "left_ptr.png"
    x_offset = 3
    y_offset = 4

  cursor = get_cursor(cursor_dir, file)

  screenshot.paste(
      cursor,
      (cursor_state.ptr_x - x_offset, cursor_state.ptr_y - y_offset),
      cursor,
  )
  return screenshot


def augment_screenshot(image: Image.Image, render_marker: bool) -> Image.Image:
  """Augments the screenshot to display additional information.

  Arguments:
    image: Should be an Image.Image.
    render_marker: Whether to render red square.

  Returns:
    Bytes corresponding to screenshot png with button press indicator.
  """
  if render_marker:
    width, _ = image.size
    draw = ImageDraw.Draw(image)
    # Render a small red rectangle in the upper right corner.
    xy = [(width - 10, 5), (width - 5, 10)]
    # RGB and transparency values from 0 to 255.
    fill_color = (255, 0, 0, 255)
    draw.rectangle(xy=xy, fill=fill_color)
    return image
  else:
    return image


def render_header(
    image: Image.Image, header: str, background_color: str
) -> Image.Image:
  """Renders a header on a PIL image and returns a new PIL image."""
  header_image = preprocessing_utils.render_text(
      header, background_color=background_color
  )
  new_width = max(header_image.width, image.width)

  new_height = int(image.height * (new_width / image.width))
  new_header_height = int(
      header_image.height * (new_width / header_image.width)
  )

  new_image = Image.new(
      "RGB", (new_width, new_height + new_header_height), "white"
  )
  new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
  new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))

  return new_image


def render_action_history(
    image: Image.Image,
    history: str,
    max_action_chars: int,
):
  history += " " * (max_action_chars - len(history))
  history = history[:max_action_chars]
  image = render_header(image, history, background_color="yellow")
  return image
