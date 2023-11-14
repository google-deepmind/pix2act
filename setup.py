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

"""setup.py for pix2act."""
import setuptools

# pylint:disable=line-too-long
setuptools.setup(
    name="pix2act",
    packages=setuptools.find_packages(),
    extras_require={
        "dev": [
            "pix2struct @ git+https://github.com/google-research/pix2struct",
            "selenium==4.0.0",
            "absl-py",
            "gin-config",
            "t5x @ git+https://github.com/google-research/t5x.git#9769393ca9f9923e17ec2f053a4b46b2192fdf53",
            "flaxformer @ git+https://github.com/google/flaxformer",
            "pycocoevalcap",
            "apache-beam",
            "Pillow",
            "grpcio",
        ],
    },
)
# pylint:enable=line-too-long
