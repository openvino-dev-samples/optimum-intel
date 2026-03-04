# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# Compatibility shim: inject _CAN_RECORD_REGISTRY and OutputRecorder into
# transformers.utils.generic if the installed transformers version doesn't have them.
# optimum.exporters.onnx._traceable_decorator imports these symbols unconditionally,
# so they must exist before that module is first imported.
import importlib as _importlib
_tug = _importlib.import_module("transformers.utils.generic")
if not hasattr(_tug, "_CAN_RECORD_REGISTRY"):
    from transformers.utils import logging as _logging
    _tug._CAN_RECORD_REGISTRY = {}
    
    class _OutputRecorder:
        def __init__(self, target_class=None, index=0, class_name=None, layer_name=None):
            self.target_class = target_class
            self.index = index
            self.class_name = class_name
            self.layer_name = layer_name
    
    _tug.OutputRecorder = _OutputRecorder
    if not hasattr(_tug, "logger"):
        _tug.logger = _logging.get_logger(__name__)

del _importlib, _tug

import optimum.exporters.openvino.model_configs

from .__main__ import main_export
from .convert import export, export_from_model, export_models, export_pytorch_via_onnx
from .stateful import ensure_stateful_is_available, patch_stateful


__all__ = ["main_export", "export", "export_models"]
