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

"""Pix2Struct tasks."""

import os

from pix2struct import tasks as pix2struct_tasks


pix2struct_tasks.add_pix2struct_task(
    name="miniwob",
    base_dir=os.environ.get("MINIWOB_DATA_DIR", ""),
    train_file_pattern="train.tfr*",
    valid_file_pattern="val.tfr*",
)

pix2struct_tasks.add_pix2struct_task(
    name="miniwob_critic",
    base_dir=os.environ.get("MINIWOB_CRITIC_DATA_DIR", ""),
    train_file_pattern="train.tfr*",
    valid_file_pattern="val.tfr*",
)

pix2struct_tasks.add_pix2struct_task(
    name="webshop",
    base_dir=os.environ.get("WEBSHOP_DATA_DIR", ""),
    train_file_pattern="train.tfr*",
    valid_file_pattern="val.tfr*",
)
