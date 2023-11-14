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

"""List of supported miniwob tasks."""

# List of tasks that are supported.
# TODO(petershaw): Reduce duplication with `eval/tasks.txt`.
SUPPORTED_TASKS = frozenset({
    "bisect-angle",
    "choose-date-nodelay",
    "circle-center",
    "click-button",
    "click-button-sequence",
    "click-checkboxes",
    "click-checkboxes-large",
    "click-checkboxes-soft",
    "click-checkboxes-transfer",
    "click-collapsible-2-nodelay",
    "click-collapsible-nodelay",
    "click-color",
    "click-dialog",
    "click-dialog-2",
    "click-link",
    "click-option",
    "click-pie-nodelay",
    "click-shades",
    "click-shape",
    "click-tab",
    "click-tab-2",
    "click-tab-2-easy",
    "click-tab-2-hard",
    "click-tab-2-medium",
    "click-test",
    "click-test-2",
    "click-test-transfer",
    "click-widget",
    "count-shape",
    "count-sides",
    "drag-box",
    "drag-item",
    "drag-items",
    "drag-items-grid",
    "drag-shapes",
    "drag-sort-numbers",
    "email-inbox-delete",
    "email-inbox-important",
    "enter-date",
    "enter-text-2",
    "enter-time",
    "find-midpoint",
    "grid-coordinate",
    "identify-shape",
    "navigate-tree",
    "number-checkboxes",
    "resize-textarea",
    "right-angle",
    "simple-algebra",
    "simple-arithmetic",
    "text-transform",
    "tic-tac-toe",
    "unicode-test",
    "use-autocomplete-nodelay",
    "use-colorwheel",
    "use-colorwheel-2",
    "use-slider",
    "use-slider-2",
    "visual-addition",
})
