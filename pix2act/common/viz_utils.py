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

"""Visualization utils."""
import base64
from typing import Any, Dict, List, Optional
from pix2act.common import tf_utils
import tensorflow as tf


def write_steps_to_html(
    steps: List[Dict[str, Any]],
    headers: List[str],
    filepath: str,
    image_scale: Optional[int] = 100,
):
  """Write examples to HTML file."""
  with tf.io.gfile.GFile(filepath, "w") as fp:
    rows = "<tr>"
    for header in headers:
      rows += f"<th>{header}</td>"
    rows += "</tr>\n"
    for step in steps:
      rows += "<tr>"
      for header in headers:
        if isinstance(step.get(header, None), bytes):
          image = base64.b64encode(step[header]).decode("ascii")
          rows += f'<td><img src="data:image/png;base64,{image}"'
          rows += f'width="{image_scale}%" height="{image_scale}%" /></td>'
        else:
          rows += f'<td>{step.get(header, "N/A")}</td>'
      rows += "</tr>\n"
    html = f"<html><body><table border='4'>{rows}</table></body></html>"
    fp.write(html)


def write_tf_examples_to_html(
    examples: List[tf.train.Example],
    filepath: str,
    image_scale: Optional[int] = 100,
):
  """Write [tf.train.Example] to HTML file."""
  steps = []
  for example in examples:
    step = {}
    step["screenshot_png"] = tf_utils.get_bytes_feature(
        example, "screenshot_png"
    )
    step["action_str"] = tf_utils.get_text_feature(example, "action_str")
    steps.append(step)
  write_steps_to_html(
      steps, ["screenshot_png", "action_str"], filepath, image_scale
  )
