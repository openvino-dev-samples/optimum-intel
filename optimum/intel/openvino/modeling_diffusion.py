#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import importlib
import inspect
import logging
import os
import shutil
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    LatentConsistencyModelImg2ImgPipeline,
    LatentConsistencyModelPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
    pipelines,
)
from diffusers.configuration_utils import ConfigMixin
from diffusers.schedulers import SchedulerMixin
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils.constants import CONFIG_NAME
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import validate_hf_hub_args
from openvino import Core
from openvino._offline_transformations import compress_model_transformation
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.modeling_outputs import ModelOutput
from transformers.utils import http_user_agent

from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)

from ...exporters.openvino import main_export
from ..utils.import_utils import is_diffusers_version
from .configuration import OVConfig, OVQuantizationConfigBase, OVQuantizationMethod, OVWeightQuantizationConfig
from .loaders import OVTextualInversionLoaderMixin
from .modeling_base import OVBaseModel, OVModelHostMixin
from .utils import (
    ONNX_WEIGHTS_NAME,
    OV_TO_PT_TYPE,
    OV_XML_FILE_NAME,
    TemporaryDirectory,
    _print_compiled_model_properties,
    classproperty,
    model_has_dynamic_inputs,
    np_to_pt_generators,
)


if is_diffusers_version(">=", "0.25.0"):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
else:
    from diffusers.models.vae import DiagonalGaussianDistribution

# Required EncoderDecoderCache object from transformers
if is_diffusers_version(">=", "0.32"):
    from diffusers import LTXPipeline
else:
    LTXPipeline = object


if is_diffusers_version(">=", "0.29.0"):
    from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline
else:
    StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline = object, object

if is_diffusers_version(">=", "0.30.0"):
    from diffusers import FluxPipeline, StableDiffusion3InpaintPipeline
else:
    StableDiffusion3InpaintPipeline = object
    FluxPipeline = object


if is_diffusers_version(">=", "0.31.0"):
    from diffusers import FluxImg2ImgPipeline, FluxInpaintPipeline
else:
    FluxImg2ImgPipeline = object
    FluxInpaintPipeline = object

if is_diffusers_version(">=", "0.32.0"):
    from diffusers import FluxFillPipeline, SanaPipeline
else:
    FluxFillPipeline = object
    SanaPipeline = object

if is_diffusers_version(">=", "0.33.0"):
    from diffusers import SanaSprintPipeline
else:
    SanaSprintPipeline = object


if is_diffusers_version(">=", "0.35.0"):
    from diffusers.models.cache_utils import CacheMixin
else:
    CacheMixin = object

try:
    from diffusers import ZImagePipeline
except ImportError:
    ZImagePipeline = object

try:
    from diffusers import ZImageOmniPipeline
except ImportError:
    ZImageOmniPipeline = object

DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER = "transformer"
DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER = "text_encoder_3"
DIFFUSION_MODEL_SIGLIP_SUBFOLDER = "siglip"

core = Core()

logger = logging.getLogger(__name__)


# TODO: support DiffusionPipeline.from_pipe()
# TODO: makes more sense to have a compositional OVMixin class
# TODO: instead of one bloated __init__, we should consider an __init__ per pipeline
class OVDiffusionPipeline(OVBaseModel, DiffusionPipeline):
    auto_model_class = DiffusionPipeline
    config_name = "model_index.json"
    _library_name = "diffusers"

    @classproperty
    def _all_ov_model_paths(cls) -> Dict[str, str]:
        models_paths = {
            "unet": os.path.join(DIFFUSION_MODEL_UNET_SUBFOLDER, OV_XML_FILE_NAME),
            "transformer": os.path.join(DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER, OV_XML_FILE_NAME),
            "vae_decoder": os.path.join(DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER, OV_XML_FILE_NAME),
            "vae_encoder": os.path.join(DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER, OV_XML_FILE_NAME),
            "text_encoder": os.path.join(DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER, OV_XML_FILE_NAME),
            "text_encoder_2": os.path.join(DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER, OV_XML_FILE_NAME),
            "text_encoder_3": os.path.join(DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER, OV_XML_FILE_NAME),
            "siglip": os.path.join(DIFFUSION_MODEL_SIGLIP_SUBFOLDER, OV_XML_FILE_NAME),
        }
        return models_paths

    def __init__(
        self,
        scheduler: SchedulerMixin,
        unet: Optional[openvino.Model] = None,
        vae_decoder: Optional[openvino.Model] = None,
        # optional pipeline models
        vae_encoder: Optional[openvino.Model] = None,
        text_encoder: Optional[openvino.Model] = None,
        text_encoder_2: Optional[openvino.Model] = None,
        text_encoder_3: Optional[openvino.Model] = None,
        transformer: Optional[openvino.Model] = None,
        siglip: Optional[openvino.Model] = None,
        # optional pipeline submodels
        tokenizer: Optional[CLIPTokenizer] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        tokenizer_3: Optional[CLIPTokenizer] = None,
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        # stable diffusion xl specific arguments
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetics_score: bool = False,
        add_watermarker: Optional[bool] = None,
        # openvino specific arguments
        device: str = "CPU",
        compile: bool = True,
        compile_only: bool = False,
        dynamic_shapes: bool = True,
        ov_config: Optional[Dict[str, str]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        quantization_config: Optional[Union[OVWeightQuantizationConfig, Dict]] = None,
        **kwargs,
    ):
        self._device = device.upper()
        self.is_dynamic = dynamic_shapes
        self._compile_only = compile_only
        self.model_save_dir = model_save_dir
        self.ov_config = {} if ov_config is None else {**ov_config}
        self.preprocessors = kwargs.get("preprocessors", [])

        if self._compile_only:
            if not compile:
                raise ValueError(
                    "`compile_only` mode does not support disabling compilation."
                    "Please provide `compile=True` if you want to use `compile_only=True` or set `compile_only=False`"
                )

            main_model = unet if unet is not None else transformer
            if not isinstance(main_model, openvino.CompiledModel):
                raise ValueError("`compile_only` expect that already compiled model will be provided")

            model_is_dynamic = model_has_dynamic_inputs(main_model)
            if dynamic_shapes ^ model_is_dynamic:
                requested_shapes = "dynamic" if dynamic_shapes else "static"
                compiled_shapes = "dynamic" if model_is_dynamic else "static"
                raise ValueError(
                    f"Provided compiled model with {compiled_shapes} shapes but requested to use {requested_shapes}. "
                    f"Please set `compile_only=False` or `dynamic_shapes={model_is_dynamic}`"
                )

        self.unet = OVModelUnet(unet, self, DIFFUSION_MODEL_UNET_SUBFOLDER) if unet is not None else None
        self.transformer = (
            OVModelTransformer(transformer, self, DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER)
            if transformer is not None
            else None
        )

        if unet is None and transformer is None:
            raise ValueError("`unet` or `transformer` model should be provided for pipeline work")
        self.vae_decoder = OVModelVaeDecoder(vae_decoder, self, DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER)
        self.vae_encoder = (
            OVModelVaeEncoder(vae_encoder, self, DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER)
            if vae_encoder is not None
            else None
        )
        self.text_encoder = (
            OVModelTextEncoder(text_encoder, self, DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER)
            if text_encoder is not None
            else None
        )
        self.text_encoder_2 = (
            OVModelTextEncoder(text_encoder_2, self, DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER)
            if text_encoder_2 is not None
            else None
        )
        self.text_encoder_3 = (
            OVModelTextEncoder(text_encoder_3, self, DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER)
            if text_encoder_3 is not None
            else None
        )
        self.siglip = (
            OVModelTextEncoder(siglip, self, DIFFUSION_MODEL_SIGLIP_SUBFOLDER)
            if siglip is not None
            else None
        )
        # We wrap the VAE Decoder & Encoder in a single object to simulate diffusers API
        self.vae = OVModelVae(decoder=self.vae_decoder, encoder=self.vae_encoder)

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.feature_extractor = feature_extractor

        # we allow passing these as torch models for now
        self.image_encoder = kwargs.pop("image_encoder", None)  # TODO: maybe mplement OVModelImageEncoder
        self.safety_checker = kwargs.pop("safety_checker", None)  # TODO: maybe mplement OVModelSafetyChecker
        self.siglip_processor = kwargs.pop("siglip_processor", None)

        all_pipeline_init_args = {
            "vae": self.vae,
            "unet": self.unet,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "text_encoder_3": self.text_encoder_3,
            "siglip": self.siglip,
            "safety_checker": self.safety_checker,
            "image_encoder": self.image_encoder,
            "siglip_processor": self.siglip_processor,
            "scheduler": self.scheduler,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "tokenizer_3": self.tokenizer_3,
            "feature_extractor": self.feature_extractor,
            "requires_aesthetics_score": requires_aesthetics_score,
            "force_zeros_for_empty_prompt": force_zeros_for_empty_prompt,
            "add_watermarker": add_watermarker,
        }

        diffusers_pipeline_args = {}
        for key in inspect.signature(self.auto_model_class).parameters.keys():
            if key in all_pipeline_init_args:
                diffusers_pipeline_args[key] = all_pipeline_init_args[key]
        # inits diffusers pipeline specific attributes (registers modules and config)
        self.auto_model_class.__init__(self, **diffusers_pipeline_args)
        # we use auto_model_class.__init__ here because we can't call super().__init__
        # as OptimizedModel already defines an __init__ which is the first in the MRO

        self._openvino_config = None
        if quantization_config:
            self._openvino_config = OVConfig(quantization_config=quantization_config)
        self._set_ov_config_parameters()

        if self.is_dynamic and not self._compile_only:
            self.reshape(batch_size=-1, height=-1, width=-1, num_images_per_prompt=-1)

        if compile and not self._compile_only:
            self.compile()

    @property
    def _component_names(self) -> List[str]:
        component_names = [name for name in self._all_ov_model_paths if getattr(self, name) is not None]
        return component_names

    @property
    def _ov_model_names(self) -> List[str]:
        return self._component_names

    @property
    def ov_models(self) -> Dict[str, openvino.Model]:
        return {name: getattr(component, "model") for name, component in self.components.items()}

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves the model to the OpenVINO IR format so that it can be re-loaded using the
        [`~optimum.intel.openvino.modeling.OVModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `Path`):
                The directory where to save the model files
        """
        if self._compile_only:
            raise ValueError(
                "`save_pretrained()` is not supported with `compile_only` mode, please initialize model without this option"
            )

        save_directory = Path(save_directory)

        models_to_save_paths = {
            component: save_directory / self._ov_model_paths[ov_component_name]
            for ov_component_name, component in self.components.items()
        }
        for model, dst_path in models_to_save_paths.items():
            save_path = dst_path.parent
            save_path.mkdir(parents=True, exist_ok=True)
            openvino.save_model(model.model, dst_path, compress_to_fp16=False)
            model_dir = (
                self.model_save_dir
                if not isinstance(self.model_save_dir, TemporaryDirectory)
                else self.model_save_dir.name
            )
            config_path = Path(model_dir) / save_path.name / CONFIG_NAME
            if config_path.is_file():
                config_save_path = save_path / CONFIG_NAME
                shutil.copyfile(config_path, config_save_path)
            else:
                if hasattr(model, "save_config"):
                    model.save_config(save_path)
                elif hasattr(model, "config") and hasattr(model.config, "save_pretrained"):
                    model.config.save_pretrained(save_path)

        self.scheduler.save_pretrained(save_directory / "scheduler")

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory / "tokenizer")
        if self.tokenizer_2 is not None:
            self.tokenizer_2.save_pretrained(save_directory / "tokenizer_2")
        if self.tokenizer_3 is not None:
            self.tokenizer_3.save_pretrained(save_directory / "tokenizer_3")
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory / "feature_extractor")
        if getattr(self, "safety_checker", None) is not None:
            self.safety_checker.save_pretrained(save_directory / "safety_checker")
        if getattr(self, "siglip_processor", None) is not None:
            self.siglip_processor.save_pretrained(save_directory / "siglip_processor")

        self._save_openvino_config(save_directory)

    def _save_config(self, save_directory):
        """
        Saves a model configuration into a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.
        """
        model_dir = (
            self.model_save_dir
            if not isinstance(self.model_save_dir, TemporaryDirectory)
            else self.model_save_dir.name
        )
        save_dir = Path(save_directory)
        original_config = Path(model_dir) / self.config_name
        if original_config.exists():
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            shutil.copy(original_config, save_dir)
        else:
            self.config.save_pretrained(save_dir)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        unet_file_name: Optional[str] = None,
        vae_decoder_file_name: Optional[str] = None,
        vae_encoder_file_name: Optional[str] = None,
        text_encoder_file_name: Optional[str] = None,
        text_encoder_2_file_name: Optional[str] = None,
        text_encoder_3_file_name: Optional[str] = None,
        transformer_file_name: Optional[str] = None,
        from_onnx: bool = False,
        load_in_8bit: bool = False,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        trust_remote_code: bool = False,
        export_model_id: Optional[str] = None,
        **kwargs,
    ):
        # same as DiffusionPipeline.from_pretraoned, if called directly, it loads the class in the config
        if cls.__name__ == "OVDiffusionPipeline":
            class_name = config["_class_name"]
            ov_pipeline_class = _get_ov_class(class_name)
        else:
            ov_pipeline_class = cls

        default_file_name = ONNX_WEIGHTS_NAME if from_onnx else OV_XML_FILE_NAME

        file_names = {
            "unet": unet_file_name or default_file_name,
            "vae_encoder": vae_encoder_file_name or default_file_name,
            "vae_decoder": vae_decoder_file_name or default_file_name,
            "text_encoder": text_encoder_file_name or default_file_name,
            "text_encoder_2": text_encoder_2_file_name or default_file_name,
            "text_encoder_3": text_encoder_3_file_name or default_file_name,
            "transformer": transformer_file_name or default_file_name,
            "siglip": default_file_name,
        }

        if not os.path.isdir(str(model_id)):
            all_components = {key for key in config.keys() if not key.startswith("_")} | {"vae_encoder", "vae_decoder"}
            allow_patterns = {os.path.join(component, "*") for component in all_components}
            allow_patterns.update(
                {
                    *file_names.values(),
                    *(file_name.replace(".xml", ".bin") for file_name in file_names.values()),
                    SCHEDULER_CONFIG_NAME,
                    cls.config_name,
                    CONFIG_NAME,
                }
            )
            ignore_patterns = ["*.msgpack", "*.safetensors", "*pytorch_model.bin"]
            if not from_onnx:
                ignore_patterns.extend(["*.onnx", "*.onnx_data"])

            model_save_folder = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                revision=revision,
                token=token,
                user_agent=http_user_agent,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        else:
            model_save_folder = str(model_id)

        model_save_path = Path(model_save_folder)

        if model_save_dir is None:
            model_save_dir = model_save_path

        submodels = {
            "scheduler": None,
            "tokenizer": None,
            "tokenizer_2": None,
            "tokenizer_3": None,
            "feature_extractor": None,
            "safety_checker": None,
            "image_encoder": None,
            "siglip_processor": None,
        }
        for name in submodels.keys():
            if name in kwargs:
                submodels[name] = kwargs.pop(name)
            elif config.get(name, (None, None))[0] is not None:
                module_name, module_class = config.get(name)
                if hasattr(pipelines, module_name):
                    module = getattr(pipelines, module_name)
                else:
                    module = importlib.import_module(module_name)
                class_obj = getattr(module, module_class)
                load_method = getattr(class_obj, "from_pretrained")
                # Check if the module is in a subdirectory
                if (model_save_path / name).is_dir():
                    submodels[name] = load_method(model_save_path / name)
                # For backward compatibility with models exported using previous optimum version, where safety_checker saving was disabled
                elif name == "safety_checker":
                    logger.warning(
                        "Pipeline config contains `safety_checker` subcomponent, while `safety_checker` is not available in model directory. "
                        "`safety_checker` will be disabled. If you want to enable it please set it explicitly to `from_pretrained` method "
                        "or reexport model with new optimum-intel version"
                    )
                    submodels[name] = None
                else:
                    submodels[name] = load_method(model_save_path)

        models = {
            ov_model_name: (model_save_path / ov_model_path).parent / file_names[ov_model_name]
            for ov_model_name, ov_model_path in cls._all_ov_model_paths.items()
        }
        for config_key, value in config.items():
            if config_key not in models and config_key not in kwargs and config_key not in submodels:
                kwargs[config_key] = value

        compile_only = kwargs.get("compile_only", False)
        if compile_only:
            ov_config = kwargs.get("ov_config", {})
            device = kwargs.get("device", "CPU")
            vae_ov_conifg = {**ov_config}
            for name, path in models.items():
                if name in kwargs:
                    models[name] = kwargs.pop(name)
                else:
                    models[name] = (
                        cls._compile_model(
                            path,
                            device,
                            ov_config if "vae" not in name else vae_ov_conifg,
                            Path(model_save_dir) / name,
                        )
                        if path.is_file()
                        else None
                    )
        else:
            for name, path in models.items():
                if name in kwargs:
                    models[name] = kwargs.pop(name)
                else:
                    models[name] = cls.load_model(path) if path.is_file() else None

        name_or_path = config.get("_name_or_path", str(model_id))
        quantization_config = quantization_config or (OVWeightQuantizationConfig(bits=8) if load_in_8bit else None)
        compile_model = kwargs.pop("compile", True)
        ov_pipeline = ov_pipeline_class(
            **models,
            **submodels,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            compile=compile_model and not quantization_config,
            **kwargs,
        )
        # same as in DiffusionPipeline.from_pretrained, we save where the model was instantiated from
        ov_pipeline.register_to_config(_name_or_path=name_or_path)

        if quantization_config:
            quantization_dataset = (
                quantization_config.dataset
                if isinstance(quantization_config, OVQuantizationConfigBase)
                else quantization_config.get("dataset", None)
            )
            if quantization_dataset is not None and ov_pipeline.export_feature != "text-to-image":
                raise NotImplementedError(
                    f"Data-aware quantization is not supported for {cls.__name__} with "
                    f"{ov_pipeline_class.export_feature} task."
                )
            # export_model_id is needed because _name_or_path is not necessarily present in the model config
            model_id = export_model_id or name_or_path
            quantization_config = cls._resolve_default_quantization_config(model_id, quantization_config)
            ov_pipeline._apply_quantization(
                quantization_config, compile_only, compile_model, model_id, trust_remote_code
            )

        return ov_pipeline

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: Dict[str, Any],
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        local_files_only: bool = False,
        load_in_8bit: Optional[bool] = None,
        quantization_config: Union[OVWeightQuantizationConfig, Dict] = None,
        compile_only: bool = False,
        **kwargs,
    ):
        if compile_only:
            logger.warning(
                "`compile_only` mode will be disabled because it does not support model export."
                "Please provide openvino model obtained using optimum-cli or saved on disk using `save_pretrained`"
            )
            compile_only = False

        # If load_in_8bit and quantization_config not specified then ov_config is set
        # to None and will be set by default in convert depending on the model size
        if load_in_8bit is None and not quantization_config:
            ov_config = None
        else:
            ov_config = OVConfig(dtype="auto")

        torch_dtype = kwargs.pop("torch_dtype", None)

        model_loading_kwargs = {}

        if torch_dtype is not None:
            model_loading_kwargs["torch_dtype"] = torch_dtype

        model_save_dir = TemporaryDirectory()
        model_save_path = Path(model_save_dir.name)
        variant = kwargs.pop("variant", None)

        main_export(
            model_name_or_path=model_id,
            output=model_save_path,
            do_validation=False,
            no_post_process=True,
            revision=revision,
            cache_dir=cache_dir,
            task=cls.export_feature,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            ov_config=ov_config,
            library_name=cls._library_name,
            variant=variant,
            model_loading_kwargs=model_loading_kwargs,
        )

        return cls._from_pretrained(
            model_id=model_save_path,
            config=config,
            from_onnx=False,
            token=token,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            load_in_8bit=load_in_8bit,
            compile_only=compile_only,
            export_model_id=model_id,  # needed to resolve default quantization config during export
            **kwargs,
        )

    def to(self, *args, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        for arg in args:
            if isinstance(arg, str):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg

        if isinstance(device, str):
            self._device = device.upper()
            self.clear_requests()
        elif device is not None:
            raise ValueError(
                "The `device` argument should be a string representing the device on which the model should be loaded."
            )

        if dtype is not None and dtype != self.dtype:
            raise NotImplementedError(
                f"Cannot change the dtype of the model from {self.dtype} to {dtype}. "
                f"Please export the model with the desired dtype."
            )

        return self

    @property
    def height(self) -> int:
        model = self.vae.decoder.model
        height = model.inputs[0].get_partial_shape()[-2]
        if height.is_dynamic:
            return -1
        return height.get_length() * (
            self.vae_scale_factor if hasattr(self, "vae_scale_factor") else self.vae_spatial_compression_ratio
        )

    @property
    def width(self) -> int:
        model = self.vae.decoder.model
        width = model.inputs[0].get_partial_shape()[-1]
        if width.is_dynamic:
            return -1
        return width.get_length() * (
            self.vae_scale_factor if hasattr(self, "vae_scale_factor") else self.vae_spatial_compression_ratio
        )

    @property
    def batch_size(self) -> int:
        model = self.unet.model if self.unet is not None else self.transformer.model
        batch_size = model.inputs[0].get_partial_shape()[0]
        if batch_size.is_dynamic:
            return -1
        return batch_size.get_length()

    def _preprocess_quantization_config(
        self,
        quantization_config: OVQuantizationConfigBase,
        model_name_or_path: str,
    ) -> OVQuantizationConfigBase:
        if isinstance(quantization_config, OVWeightQuantizationConfig) and quantization_config.dataset is not None:
            quantization_config = quantization_config.clone()
            quantization_config.quant_method = OVQuantizationMethod.HYBRID
        return quantization_config

    def _reshape_unet(
        self,
        model: openvino.Model,
        batch_size: int = -1,
        height: int = -1,
        width: int = -1,
        num_images_per_prompt: int = -1,
        tokenizer_max_length: int = -1,
    ):
        if batch_size == -1 or num_images_per_prompt == -1:
            batch_size = -1
        else:
            batch_size *= num_images_per_prompt
            # The factor of 2 comes from the guidance scale > 1
            if "timestep_cond" not in {inputs.get_any_name() for inputs in model.inputs}:
                batch_size *= 2

        height = height // self.vae_scale_factor if height > 0 else height
        width = width // self.vae_scale_factor if width > 0 else width
        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            if inputs.get_any_name() == "timestep":
                if shapes[inputs].rank == 1:
                    shapes[inputs][0] = 1
            elif inputs.get_any_name() == "sample":
                in_channels = self.unet.config.get("in_channels", None)
                if in_channels is None:
                    in_channels = shapes[inputs][1]
                    if in_channels.is_dynamic:
                        logger.warning(
                            "Could not identify `in_channels` from the unet configuration, to statically reshape the unet please provide a configuration."
                        )
                        self.is_dynamic = True

                shapes[inputs] = [batch_size, in_channels, height, width]
            elif inputs.get_any_name() == "text_embeds":
                shapes[inputs] = [batch_size, self.text_encoder_2.config["projection_dim"]]
            elif inputs.get_any_name() == "time_ids":
                shapes[inputs] = [batch_size, inputs.get_partial_shape()[1]]
            elif inputs.get_any_name() == "timestep_cond":
                shapes[inputs] = [batch_size, self.unet.config["time_cond_proj_dim"]]
            else:
                shapes[inputs][0] = batch_size
                shapes[inputs][1] = tokenizer_max_length
        model.reshape(shapes)
        return model

    def _reshape_transformer(
        self,
        model: openvino.Model,
        batch_size: int = -1,
        height: int = -1,
        width: int = -1,
        num_images_per_prompt: int = -1,
        tokenizer_max_length: int = -1,
        num_frames: int = -1,
    ):
        if batch_size == -1 or num_images_per_prompt == -1:
            batch_size = -1
        else:
            # The factor of 2 comes from the guidance scale > 1
            batch_size *= num_images_per_prompt
            if "img_ids" not in {inputs.get_any_name() for inputs in model.inputs}:
                batch_size *= 2

        is_ltx = self.__class__.__name__.startswith("OVLTX")
        if is_ltx:
            height = height // self.vae_spatial_compression_ratio if height > 0 else -1
            width = width // self.vae_spatial_compression_ratio if width > 0 else -1
            packed_height_width = width * height * num_frames if height > 0 and width > 0 and num_frames > 0 else -1
        else:
            height = height // self.vae_scale_factor if height > 0 else height
            width = width // self.vae_scale_factor if width > 0 else width
            packed_height = height // 2 if height > 0 else height
            packed_width = width // 2 if width > 0 else width
            packed_height_width = packed_width * packed_height if height > 0 and width > 0 else -1

        shapes = {}
        for inputs in model.inputs:
            shapes[inputs] = inputs.get_partial_shape()
            if inputs.get_any_name() in ["timestep", "guidance"]:
                shapes[inputs][0] = batch_size
            elif inputs.get_any_name() == "hidden_states":
                in_channels = self.transformer.config.get("in_channels", None)
                if in_channels is None:
                    in_channels = (
                        shapes[inputs][1] if inputs.get_partial_shape().rank.get_length() == 4 else shapes[inputs][2]
                    )
                    if in_channels.is_dynamic:
                        logger.warning(
                            "Could not identify `in_channels` from the unet configuration, to statically reshape the unet please provide a configuration."
                        )
                        self.is_dynamic = True
                if inputs.get_partial_shape().rank.get_length() == 4:
                    shapes[inputs] = [batch_size, in_channels, height, width]
                else:
                    shapes[inputs] = [batch_size, packed_height_width, in_channels]

            elif inputs.get_any_name() == "pooled_projections":
                shapes[inputs] = [batch_size, self.transformer.config["pooled_projection_dim"]]
            elif inputs.get_any_name() == "img_ids":
                shapes[inputs] = (
                    [batch_size, packed_height_width, 3]
                    if is_diffusers_version("<", "0.31.0")
                    else [packed_height_width, 3]
                )
            elif inputs.get_any_name() == "txt_ids":
                shapes[inputs] = [batch_size, -1, 3] if is_diffusers_version("<", "0.31.0") else [-1, 3]
            elif inputs.get_any_name() in ["height", "width", "num_frames", "rope_interpolation_scale"]:
                shapes[inputs] = inputs.get_partial_shape()
            else:
                shapes[inputs][0] = batch_size
                shapes[inputs][1] = -1  # text_encoder_3 may have vary input length
        model.reshape(shapes)
        return model

    def _reshape_text_encoder(self, model: openvino.Model, batch_size: int = -1, tokenizer_max_length: int = -1):
        if batch_size != -1:
            shapes = {input_tensor: [batch_size, tokenizer_max_length] for input_tensor in model.inputs}
            model.reshape(shapes)
        return model

    def _reshape_vae_encoder(
        self,
        model: openvino.Model,
        batch_size: int = -1,
        height: int = -1,
        width: int = -1,
        num_frames: int = -1,
    ):
        in_channels = self.vae_encoder.config.get("in_channels", None)
        if in_channels is None:
            in_channels = model.inputs[0].get_partial_shape()[1]
            if in_channels.is_dynamic:
                logger.warning(
                    "Could not identify `in_channels` from the VAE encoder configuration, to statically reshape the VAE encoder please provide a configuration."
                )
                self.is_dynamic = True
        shapes = {
            model.inputs[0]: [batch_size, in_channels, height, width]
            if model.inputs[0].get_partial_shape().rank.get_length() == 4
            else [batch_size, in_channels, num_frames, height, width]
        }
        model.reshape(shapes)
        return model

    def _reshape_vae_decoder(
        self,
        model: openvino.Model,
        height: int = -1,
        width: int = -1,
        num_images_per_prompt: int = -1,
        num_frames: int = -1,
    ):
        is_ltx = self.__class__.__name__.startswith("OVLTX")
        if is_ltx:
            height = height // self.vae_spatial_compression_ratio if height > 0 else -1
            width = width // self.vae_spatial_compression_ratio if width > 0 else -1
        else:
            height = height // self.vae_scale_factor if height > -1 else height
            width = width // self.vae_scale_factor if width > -1 else width
        latent_channels = self.vae_decoder.config.get("latent_channels", None)
        if latent_channels is None:
            latent_channels = model.inputs[0].get_partial_shape()[1]
            if latent_channels.is_dynamic:
                logger.warning(
                    "Could not identify `latent_channels` from the VAE decoder configuration, to statically reshape the VAE decoder please provide a configuration."
                )
                self.is_dynamic = True
        shapes = {
            model.inputs[0]: [num_images_per_prompt, latent_channels, height, width]
            if not is_ltx
            else [num_images_per_prompt, latent_channels, num_frames, height, width]
        }
        model.reshape(shapes)
        return model

    def reshape(self, batch_size: int, height: int, width: int, num_images_per_prompt: int = -1, num_frames: int = -1):
        if self._compile_only:
            raise ValueError(
                "`reshape()` is not supported with `compile_only` mode, please initialize model without this option"
            )

        self.is_dynamic = -1 in {batch_size, height, width, num_images_per_prompt}

        if self.tokenizer is None and self.tokenizer_2 is None:
            tokenizer_max_len = -1
        else:
            if self.tokenizer is not None and "Gemma" in self.tokenizer.__class__.__name__:
                tokenizer_max_len = -1
            else:
                tokenizer_max_len = (
                    getattr(self.tokenizer, "model_max_length", -1)
                    if self.tokenizer is not None
                    else getattr(self.tokenizer_2, "model_max_length", -1)
                )

        if self.unet is not None:
            self.unet.model = self._reshape_unet(
                self.unet.model, batch_size, height, width, num_images_per_prompt, tokenizer_max_len
            )
        if self.transformer is not None:
            self.transformer.model = self._reshape_transformer(
                self.transformer.model,
                batch_size,
                height,
                width,
                num_images_per_prompt,
                tokenizer_max_len,
                num_frames=num_frames,
            )
        self.vae_decoder.model = self._reshape_vae_decoder(
            self.vae_decoder.model, height, width, num_images_per_prompt, num_frames=num_frames
        )

        if self.vae_encoder is not None:
            self.vae_encoder.model = self._reshape_vae_encoder(
                self.vae_encoder.model, batch_size, height, width, num_frames=num_frames
            )

        if self.text_encoder is not None:
            self.text_encoder.model = self._reshape_text_encoder(
                # GemmaTokenizer uses inf as model_max_length, Text Encoder in LTX do not pad input to model_max_length
                self.text_encoder.model,
                batch_size,
                (
                    getattr(self.tokenizer, "model_max_length", -1)
                    if "Gemma" not in self.tokenizer.__class__.__name__
                    and not self.__class__.__name__.startswith("OVLTX")
                    else -1
                ),
            )

        if self.text_encoder_2 is not None:
            self.text_encoder_2.model = self._reshape_text_encoder(
                self.text_encoder_2.model, batch_size, getattr(self.tokenizer_2, "model_max_length", -1)
            )

        if self.text_encoder_3 is not None:
            self.text_encoder_3.model = self._reshape_text_encoder(self.text_encoder_3.model, batch_size, -1)

        self.clear_requests()
        return self

    def half(self):
        """
        Converts all the model weights to FP16 for more efficient inference on GPU.
        """
        if self._compile_only:
            raise ValueError(
                "`half()` is not supported with `compile_only` mode, please initialize model without this option"
            )

        for ov_model in self.ov_models.values():
            compress_model_transformation(ov_model)

        self.clear_requests()

        return self

    def clear_requests(self):
        if self._compile_only:
            raise ValueError(
                "`clear_requests()` is not supported with `compile_only` mode, please initialize model without this option"
            )
        for component in self.components.values():
            component.clear_requests()

    def compile(self):
        for component in self.components.values():
            component.compile()

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def __call__(self, *args, **kwargs):
        # we do this to keep numpy random states support for now
        # TODO: deprecate and add warnings when a random state is passed

        args = list(args)
        for i in range(len(args)):
            args[i] = np_to_pt_generators(args[i], self.device)

        for k, v in kwargs.items():
            kwargs[k] = np_to_pt_generators(v, self.device)

        height, width = None, None
        height_idx, width_idx = None, None
        shapes_overridden = False
        sig = inspect.signature(self.auto_model_class.__call__)
        sig_height_idx = list(sig.parameters).index("height") if "height" in sig.parameters else len(sig.parameters)
        sig_width_idx = list(sig.parameters).index("width") if "width" in sig.parameters else len(sig.parameters)
        if "height" in kwargs:
            height = kwargs["height"]
        elif len(args) > sig_height_idx:
            height = args[sig_height_idx]
            height_idx = sig_height_idx

        if "width" in kwargs:
            width = kwargs["width"]
        elif len(args) > sig_width_idx:
            width = args[sig_width_idx]
            width_idx = sig_width_idx

        if self.height != -1:
            if height is not None and height != self.height:
                logger.warning(f"Incompatible height argument provided {height}. Pipeline only support {self.height}.")
                height = self.height
            else:
                height = self.height

            if height_idx is not None:
                args[height_idx] = height
            else:
                kwargs["height"] = height

            shapes_overridden = True

        if self.width != -1:
            if width is not None and width != self.width:
                logger.warning(f"Incompatible widtth argument provided {width}. Pipeline only support {self.width}.")
                width = self.width
            else:
                width = self.width

            if width_idx is not None:
                args[width_idx] = width
            else:
                kwargs["width"] = width
            shapes_overridden = True

        # Sana generates images in specific resolution grid size and then resize to requested size by default, it may contradict with pipeline height / width
        # Disable this behavior for static shape pipeline
        if self.auto_model_class.__name__.startswith("Sana") and shapes_overridden:
            sig_resolution_bining_idx = (
                list(sig.parameters).index("use_resolution_binning")
                if "use_resolution_binning" in sig.parameters
                else len(sig.parameters)
            )
            if len(args) > sig_resolution_bining_idx:
                args[sig_resolution_bining_idx] = False
            else:
                kwargs["use_resolution_binning"] = False
        # we use auto_model_class.__call__ here because we can't call super().__call__
        # as OptimizedModel already defines a __call__ which is the first in the MRO
        return self.auto_model_class.__call__(self, *args, **kwargs)


class OVPipelinePart(OVModelHostMixin, ConfigMixin, CacheMixin):
    config_name: str = CONFIG_NAME

    def __init__(
        self,
        model: openvino.Model,
        parent_pipeline: OVDiffusionPipeline,
        model_name: str = "",
    ):
        self.model = model
        self.model_name = model_name
        self.parent_pipeline = parent_pipeline
        self.request = None if not parent_pipeline._compile_only else self.model
        self.ov_config = parent_pipeline.ov_config

        if isinstance(parent_pipeline.model_save_dir, TemporaryDirectory):
            self.model_save_dir = Path(parent_pipeline.model_save_dir.name) / self.model_name
        else:
            self.model_save_dir = Path(parent_pipeline.model_save_dir) / self.model_name

        config_file_path = self.model_save_dir / self.config_name

        if not config_file_path.is_file():
            # config is mandatory for the model part to be used for inference
            raise ValueError(f"Configuration file for {self.__class__.__name__} not found at {config_file_path}")

        config_dict = self._dict_from_json_file(config_file_path)
        self.register_to_config(**config_dict)

    @property
    def _device(self) -> str:
        return self.parent_pipeline._device

    @property
    def device(self) -> torch.device:
        return self.parent_pipeline.device

    @property
    def dtype(self) -> torch.dtype:
        return OV_TO_PT_TYPE[self.ov_config.get("dtype", "f32")]

    def clear_requests(self):
        if self.parent_pipeline._compile_only:
            raise ValueError(
                "`clear_requests()` is not supported with `compile_only` mode, please initialize model without this option"
            )
        self.request = None

    def compile(self):
        if self.request is None:
            if (
                "CACHE_DIR" not in self.ov_config.keys()
                and not str(self.model_save_dir).startswith(gettempdir())
                and "GPU" in self._device
            ):
                self.ov_config["CACHE_DIR"] = os.path.join(self.model_save_dir, "model_cache")

            logger.info(f"Compiling the {self.model_name} to {self._device} ...")
            self.request = core.compile_model(self.model, self._device, self.ov_config)
            # OPENVINO_LOG_LEVEL can be found in https://docs.openvino.ai/2023.2/openvino_docs_OV_UG_supported_plugins_AUTO_debugging.html
            if "OPENVINO_LOG_LEVEL" in os.environ and int(os.environ["OPENVINO_LOG_LEVEL"]) > 2:
                _print_compiled_model_properties(self.request)

    def to(self, *args, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        for arg in args:
            if isinstance(arg, str):
                device = arg
            elif isinstance(arg, torch.dtype):
                dtype = arg

        if isinstance(device, str):
            self._device = device.upper()
            self.request = None
        elif device is not None:
            raise ValueError(
                "The `device` argument should be a string representing the device on which the model should be loaded."
            )

        if dtype is not None and dtype != self.dtype:
            raise NotImplementedError(
                f"Cannot change the dtype of the model from {self.dtype} to {dtype}. "
                f"Please export the model with the desired dtype."
            )

        return self

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def modules(self):
        return []

    def named_modules(self):
        # starting from diffusers 0.35.0 some model parts inherit from `CacheMixin` which uses `named_modules` method
        # to register some hooks for attention caching, we return empty list here since it can't be used with OpenVINO
        yield from []


class OVModelTextEncoder(OVPipelinePart):
    def __init__(self, model: openvino.Model, parent_pipeline: OVDiffusionPipeline, model_name: str = ""):
        super().__init__(model, parent_pipeline, model_name)
        self.hidden_states_output_names = [
            name for out in self.model.outputs for name in out.names if name.startswith("hidden_states")
        ]
        self.input_names = [inp.get_any_name() for inp in self.model.inputs]

    def forward(
        self,
        input_ids: Union[np.ndarray, torch.Tensor],
        attention_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = False,
    ):
        self.compile()
        model_inputs = {"input_ids": input_ids}

        if "attention_mask" in self.input_names:
            model_inputs["attention_mask"] = attention_mask

        ov_outputs = self.request(model_inputs, share_inputs=True)
        main_out = ov_outputs[0]
        model_outputs = {}
        model_outputs[self.model.outputs[0].get_any_name()] = torch.from_numpy(main_out)
        if len(self.model.outputs) > 1 and "pooler_output" in self.model.outputs[1].get_any_name():
            model_outputs["pooler_output"] = torch.from_numpy(ov_outputs[1])
        if self.hidden_states_output_names and "last_hidden_state" not in model_outputs:
            model_outputs["last_hidden_state"] = torch.from_numpy(ov_outputs[self.hidden_states_output_names[-1]])
        if (
            self.hidden_states_output_names
            and output_hidden_states
            or getattr(self.config, "output_hidden_states", False)
        ):
            hidden_states = [torch.from_numpy(ov_outputs[out_name]) for out_name in self.hidden_states_output_names]
            model_outputs["hidden_states"] = hidden_states

        if return_dict:
            return model_outputs
        return ModelOutput(**model_outputs)


class OVModelUnet(OVPipelinePart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self.config, "time_cond_proj_dim"):
            logger.warning(
                "The `time_cond_proj_dim` attribute is missing from the UNet configuration. "
                "Please re-export the model with newer version of optimum and diffusers."
            )
            self.register_to_config(time_cond_proj_dim=None)

    def forward(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        timestep: Union[np.ndarray, torch.Tensor],
        encoder_hidden_states: Union[np.ndarray, torch.Tensor],
        text_embeds: Optional[Union[np.ndarray, torch.Tensor]] = None,
        time_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        timestep_cond: Optional[Union[np.ndarray, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ):
        self.compile()

        model_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        if text_embeds is not None:
            model_inputs["text_embeds"] = text_embeds
        if time_ids is not None:
            model_inputs["time_ids"] = time_ids
        if timestep_cond is not None:
            model_inputs["timestep_cond"] = timestep_cond
        if cross_attention_kwargs is not None:
            model_inputs.update(cross_attention_kwargs)
        if added_cond_kwargs is not None:
            model_inputs.update(added_cond_kwargs)

        ov_outputs = self.request(model_inputs, share_inputs=True).to_dict()

        model_outputs = {}
        for key, value in ov_outputs.items():
            model_outputs[next(iter(key.names))] = torch.from_numpy(value)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class OVModelTransformer(OVPipelinePart):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: torch.LongTensor = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        video_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        self.compile()

        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        if pooled_projections is not None:
            model_inputs["pooled_projections"] = pooled_projections
        if img_ids is not None:
            model_inputs["img_ids"] = img_ids
        if txt_ids is not None:
            model_inputs["txt_ids"] = txt_ids
        if guidance is not None:
            model_inputs["guidance"] = guidance

        if encoder_attention_mask is not None:
            model_inputs["encoder_attention_mask"] = encoder_attention_mask
        if num_frames is not None:
            model_inputs["num_frames"] = num_frames
        if height is not None:
            model_inputs["height"] = height
        if width is not None:
            model_inputs["width"] = width
        if rope_interpolation_scale is not None:
            if not isinstance(rope_interpolation_scale, torch.Tensor):
                rope_interpolation_scale = torch.tensor(rope_interpolation_scale)
            model_inputs["rope_interpolation_scale"] = rope_interpolation_scale

        ov_outputs = self.request(model_inputs, share_inputs=True).to_dict()

        model_outputs = {}
        for key, value in ov_outputs.items():
            model_outputs[next(iter(key.names))] = torch.from_numpy(value)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class OVModelVaeEncoder(OVPipelinePart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self.config, "scaling_factor"):
            logger.warning(
                "The `scaling_factor` attribute is missing from the VAE encoder configuration. "
                "Please re-export the model with newer version of optimum and diffusers."
            )
            self.register_to_config(scaling_factor=2 ** (len(self.config.block_out_channels) - 1))

    def forward(
        self,
        sample: Union[np.ndarray, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ):
        self.compile()

        model_inputs = {"sample": sample}

        ov_outputs = self.request(model_inputs, share_inputs=True).to_dict()

        model_outputs = {}
        for key, value in ov_outputs.items():
            model_outputs[next(iter(key.names))] = torch.from_numpy(value)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")

        if "latent_parameters" in model_outputs:
            model_outputs["latent_dist"] = DiagonalGaussianDistribution(
                parameters=model_outputs.pop("latent_parameters")
            )

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class OVModelVaeDecoder(OVPipelinePart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # can be missing from models exported long ago
        if not hasattr(self.config, "scaling_factor"):
            logger.warning(
                "The `scaling_factor` attribute is missing from the VAE decoder configuration. "
                "Please re-export the model with newer version of optimum and diffusers."
            )
            self.register_to_config(scaling_factor=2 ** (len(self.config.block_out_channels) - 1))

    def forward(
        self,
        latent_sample: Union[np.ndarray, torch.Tensor],
        timestep: Optional[Union[np.ndarray, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ):
        self.compile()

        model_inputs = {"latent_sample": latent_sample}

        if timestep is not None:
            model_inputs["timestep"] = timestep

        ov_outputs = self.request(model_inputs, share_inputs=True).to_dict()

        model_outputs = {}
        for key, value in ov_outputs.items():
            model_outputs[next(iter(key.names))] = torch.from_numpy(value)

        if return_dict:
            return model_outputs

        return ModelOutput(**model_outputs)


class OVModelVae(OVModelHostMixin):
    def __init__(self, decoder: OVModelVaeDecoder, encoder: OVModelVaeEncoder):
        self.decoder = decoder
        self.encoder = encoder
        self.spatial_compression_ratio, self.temporal_compression_ratio = None, None
        if hasattr(self.decoder.config, "spatio_temporal_scaling"):
            patch_size = self.decoder.config.patch_size
            patch_size_t = self.decoder.config.patch_size_t
            spatio_temporal_scaling = self.decoder.config.spatio_temporal_scaling
            self.spatial_compression_ratio = patch_size * 2 ** sum(spatio_temporal_scaling)
            self.temporal_compression_ratio = patch_size_t * 2 ** sum(spatio_temporal_scaling)
        self.latents_mean, self.latents_std = None, None
        if hasattr(self.decoder.config, "latents_mean_data"):
            self.latents_mean = torch.tensor(self.decoder.config.latents_mean_data)
        if hasattr(self.decoder.config, "latents_std_data"):
            self.latents_std = torch.tensor(self.decoder.config.latents_std_data)

    @property
    def _component_names(self) -> List[str]:
        return ["encoder", "decoder"]

    @property
    def _ov_model_names(self) -> List[str]:
        return self._component_names

    @property
    def ov_models(self) -> Dict[str, Union[openvino.Model, openvino.CompiledModel]]:
        return {name: getattr(component, "model") for name, component in self.components.items()}

    @property
    def config(self):
        return self.decoder.config

    @property
    def dtype(self):
        return self.decoder.dtype

    @property
    def device(self):
        return self.decoder.device

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.decoder.to(*args, **kwargs)
        if self.encoder is not None:
            self.encoder.to(*args, **kwargs)


class OVStableDiffusionPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion#diffusers.StableDiffusionPipeline).
    """

    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = StableDiffusionPipeline


class OVStableDiffusionImg2ImgPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionImg2ImgPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_img2img#diffusers.StableDiffusionImg2ImgPipeline).
    """

    main_input_name = "image"
    export_feature = "image-to-image"
    auto_model_class = StableDiffusionImg2ImgPipeline


class OVStableDiffusionInpaintPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionInpaintPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_inpaint#diffusers.StableDiffusionInpaintPipeline).
    """

    main_input_name = "image"
    export_feature = "inpainting"
    auto_model_class = StableDiffusionInpaintPipeline


class OVStableDiffusionXLPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionXLPipeline):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline).
    """

    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = StableDiffusionXLPipeline

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


class OVStableDiffusionXLImg2ImgPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionXLImg2ImgPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline).
    """

    main_input_name = "image"
    export_feature = "image-to-image"
    auto_model_class = StableDiffusionXLImg2ImgPipeline

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids


class OVStableDiffusionXLInpaintPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusionXLInpaintPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.StableDiffusionXLInpaintPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLInpaintPipeline).
    """

    main_input_name = "image"
    export_feature = "inpainting"
    auto_model_class = StableDiffusionXLInpaintPipeline

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids


class OVLatentConsistencyModelPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, LatentConsistencyModelPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency#diffusers.LatentConsistencyModelPipeline).
    """

    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = LatentConsistencyModelPipeline


class OVLatentConsistencyModelImg2ImgPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, LatentConsistencyModelImg2ImgPipeline
):
    """
    OpenVINO-powered stable diffusion pipeline corresponding to [diffusers.LatentConsistencyModelImg2ImgPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_consistency_img2img#diffusers.LatentConsistencyModelImg2ImgPipeline).
    """

    main_input_name = "image"
    export_feature = "image-to-image"
    auto_model_class = LatentConsistencyModelImg2ImgPipeline


class OVStableDiffusion3Pipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusion3Pipeline):
    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = StableDiffusion3Pipeline


class OVStableDiffusion3Img2ImgPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusion3Img2ImgPipeline
):
    main_input_name = "image"
    export_feature = "image-to-image"
    auto_model_class = StableDiffusion3Img2ImgPipeline


class OVStableDiffusion3InpaintPipeline(
    OVDiffusionPipeline, OVTextualInversionLoaderMixin, StableDiffusion3InpaintPipeline
):
    main_input_name = "image"
    export_feature = "inpainting"
    auto_model_class = StableDiffusion3InpaintPipeline


class OVFluxPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, FluxPipeline):
    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = FluxPipeline


class OVFluxImg2ImgPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, FluxImg2ImgPipeline):
    main_input_name = "image"
    export_feature = "image-to-image"
    auto_model_class = FluxImg2ImgPipeline


class OVFluxInpaintPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, FluxInpaintPipeline):
    main_input_name = "image"
    export_feature = "inpainting"
    auto_model_class = FluxInpaintPipeline


class OVFluxFillPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, FluxFillPipeline):
    main_input_name = "image"
    export_feature = "inpainting"
    auto_model_class = FluxFillPipeline


class OVSanaPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, SanaPipeline):
    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = SanaPipeline


class OVSanaSprintPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, SanaSprintPipeline):
    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = SanaSprintPipeline


class OVLTXPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, LTXPipeline):
    main_input_name = "prompt"
    export_feature = "text-to-video"
    auto_model_class = LTXPipeline


class OVZImagePipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, ZImagePipeline):
    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = ZImagePipeline

    def reshape(self, batch_size: int, height: int, width: int, num_images_per_prompt: int = -1, num_frames: int = -1):
        # Z-Image transformer uses 5D inputs with hardcoded spatial reshapes from tracing.
        # Only reshape the text_encoder (dynamic seq_len); keep transformer and VAE static.
        self.is_dynamic = False
        if self.text_encoder is not None:
            self._reshape_text_encoder(self.text_encoder.model, batch_size=-1, tokenizer_max_length=-1)
        return self

    def _call_ov_transformer(self, x_list, t, cap_feats_list):
        """Call OV transformer per-batch-item (model was traced with batch=1)."""
        self.transformer.compile()
        results = []
        for i in range(len(x_list)):
            hidden_states = x_list[i].unsqueeze(0).to(torch.float32)
            encoder_hidden_states = cap_feats_list[i].unsqueeze(0).to(torch.float32)
            timestep = t[i:i+1].to(torch.float32)

            model_inputs = {
                "hidden_states": hidden_states,
                "encoder_hidden_states": encoder_hidden_states,
                "timestep": timestep,
            }
            ov_outputs = self.transformer.request(model_inputs, share_inputs=True).to_dict()
            result_tensor = torch.from_numpy(next(iter(ov_outputs.values())))
            results.append(result_tensor)
        return results

    def _encode_prompt(self, prompt, device=None, prompt_embeds=None, max_sequence_length=512):
        """Encode prompt using OV text encoder. Returns list of per-prompt embeddings."""
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [{"role": "user", "content": prompt_item}]
            prompt[i] = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )

        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        # OV text encoder outputs hidden_states[-2] as last_hidden_state
        te_output = self.text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_masks,
        )
        prompt_embeds = te_output.last_hidden_state

        embeddings_list = []
        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        sigmas=None,
        guidance_scale=5.0,
        cfg_normalization=False,
        cfg_truncation=1.0,
        negative_prompt=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        joint_attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    ):
        from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, retrieve_timesteps
        from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput

        height = height or 1024
        width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(f"Height must be divisible by {vae_scale} (got {height}).")
        if width % vae_scale != 0:
            raise ValueError(f"Width must be divisible by {vae_scale} (got {width}).")

        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        if prompt_embeds is None or prompt is not None:
            prompt_embeds = self._encode_prompt(
                prompt=prompt, device=device, max_sequence_length=max_sequence_length,
            )
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = self._encode_prompt(
                    prompt=negative_prompt or [""] * batch_size, device=device,
                    max_sequence_length=max_sequence_length,
                )
            else:
                negative_prompt_embeds = []

        num_channels_latents = self.transformer.config.get("in_channels", 16)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000
                t_norm = timestep[0].item()

                current_guidance_scale = self.guidance_scale
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    if t_norm > self._cfg_truncation:
                        current_guidance_scale = 0.0

                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

                if apply_cfg:
                    latent_model_input = latents.repeat(2, 1, 1, 1)
                    prompt_embeds_input = prompt_embeds + negative_prompt_embeds
                    timestep_input = timestep.repeat(2)
                else:
                    latent_model_input = latents
                    prompt_embeds_input = prompt_embeds
                    timestep_input = timestep

                latent_model_input = latent_model_input.unsqueeze(2)
                x_list = list(latent_model_input.unbind(dim=0))

                model_out_list = self._call_ov_transformer(x_list, timestep_input, prompt_embeds_input)

                if apply_cfg:
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]

                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()
                        pred = pos + current_guidance_scale * (pos - neg)
                        if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(pred)
                            max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)
                        noise_pred.append(pred)
                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([item.float() for item in model_out_list], dim=0)

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred

                latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in (callback_on_step_end_tensor_inputs or []):
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return ZImagePipelineOutput(images=image)


class OVZImageOmniPipeline(OVDiffusionPipeline, OVTextualInversionLoaderMixin, ZImageOmniPipeline):
    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = ZImageOmniPipeline

    # Split transformer sub-model names
    _SPLIT_TRANSFORMER_PARTS = [
        "transformer_patch_embed",
        "transformer_noise_refiner",
        "transformer_context_refiner",
        "transformer_siglip_refiner",
        "transformer_main",
    ]

    SEQ_MULTI_OF = 32  # Padding alignment from original transformer

    def reshape(self, batch_size: int, height: int, width: int, num_images_per_prompt: int = -1, num_frames: int = -1):
        self.is_dynamic = False
        if self.text_encoder is not None:
            self._reshape_text_encoder(self.text_encoder.model, batch_size=-1, tokenizer_max_length=-1)
        return self

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        """Override to detect and load split transformer sub-models."""
        model_path = Path(model_id)

        # Check if this is a split transformer layout
        has_split = all((model_path / part / "openvino_model.xml").exists() for part in cls._SPLIT_TRANSFORMER_PARTS)

        if has_split:
            return cls._from_pretrained_split(model_path, **kwargs)
        else:
            # Fall back to standard loading (with export support)
            return super().from_pretrained(model_id, **kwargs)

    @classmethod
    def _from_pretrained_split(cls, model_path, **kwargs):
        """Load pipeline with split transformer sub-models."""
        import openvino as ov
        import json

        device = kwargs.pop("device", "CPU")
        compile_model = kwargs.pop("compile", True)
        ov_config = kwargs.pop("ov_config", None) or {}

        core = ov.Core()

        # Load split transformer sub-models
        split_models = {}
        for part_name in cls._SPLIT_TRANSFORMER_PARTS:
            xml_path = model_path / part_name / "openvino_model.xml"
            split_models[part_name] = core.read_model(str(xml_path))

        # Load other standard components
        std_components = {}
        for comp_name in ["text_encoder", "vae_decoder", "vae_encoder", "siglip"]:
            xml_path = model_path / comp_name / "openvino_model.xml"
            if xml_path.exists():
                std_components[comp_name] = core.read_model(str(xml_path))
            else:
                std_components[comp_name] = None

        # Load submodels (scheduler, tokenizer, etc.)
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(model_path / "scheduler"))

        tokenizer = None
        tokenizer_path = model_path / "tokenizer"
        if tokenizer_path.exists():
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        siglip_processor = None
        siglip_proc_path = model_path / "siglip_processor"
        if siglip_proc_path.exists():
            from transformers import AutoImageProcessor
            siglip_processor = AutoImageProcessor.from_pretrained(str(siglip_proc_path))

        # Load transformer config
        transformer_config = {}
        config_path = model_path / "transformer" / "config.json"
        if not config_path.exists():
            config_path = model_path / "transformer_patch_embed" / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                transformer_config = json.load(f)

        # We need a "transformer" OV model for the base class constructor.
        # Use the main transformer as a stand-in (it won't be called directly).
        # The base __init__ requires a transformer parameter.
        pipeline = cls(
            scheduler=scheduler,
            transformer=split_models["transformer_main"],
            vae_decoder=std_components["vae_decoder"],
            vae_encoder=std_components.get("vae_encoder"),
            text_encoder=std_components.get("text_encoder"),
            siglip=std_components.get("siglip"),
            tokenizer=tokenizer,
            siglip_processor=siglip_processor,
            device=device,
            compile=False,
            dynamic_shapes=True,
            ov_config=ov_config,
            model_save_dir=str(model_path),
        )

        # Store split models and compile them
        pipeline._split_models = {}
        pipeline._split_compiled = {}
        pipeline._ov_core = core
        pipeline._ov_device = device
        pipeline._ov_config = ov_config
        for part_name in cls._SPLIT_TRANSFORMER_PARTS:
            pipeline._split_models[part_name] = split_models[part_name]
            pipeline._split_compiled[part_name] = None

        # Store transformer config for Python-side logic
        pipeline._transformer_config = transformer_config

        # Initialize RoPE embedder for Python-side position encoding
        pipeline._init_rope_embedder(transformer_config)

        if compile_model:
            pipeline.compile()

        return pipeline

    def _init_rope_embedder(self, config):
        """Initialize RoPE embedder with precomputed frequencies for Python-side use."""
        axes_dims = config.get("axes_dims", [32, 48, 48])
        axes_lens = config.get("axes_lens", [1536, 512, 512])
        theta = config.get("rope_theta", 256.0)

        freqs_cis = []
        for d, e in zip(axes_dims, axes_lens):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            timestep = torch.arange(e, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            cos_vals = torch.cos(freqs).repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)
            sin_vals = torch.sin(freqs).repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)
            freqs_cis.append(torch.stack([cos_vals, sin_vals], dim=-1))  # [max_len, dim, 2]

        self._rope_freqs_cis = freqs_cis
        self._axes_dims = axes_dims

    def _rope_embed(self, ids):
        """Compute RoPE embeddings from position IDs. ids: [N, 3]"""
        result = []
        device = ids.device
        for i in range(len(self._axes_dims)):
            index = ids[:, i]
            fc = self._rope_freqs_cis[i].to(device)
            result.append(fc[index])
        return torch.cat(result, dim=-2)  # [N, total_dim, 2]

    def _get_split_compiled(self, part_name):
        """Get or compile a split sub-model."""
        if self._split_compiled.get(part_name) is None:
            self._split_compiled[part_name] = self._ov_core.compile_model(
                self._split_models[part_name], self._ov_device, self._ov_config
            )
        return self._split_compiled[part_name]

    def _call_split_model(self, part_name, inputs):
        """Call a split OV sub-model with named inputs dict."""
        compiled = self._get_split_compiled(part_name)
        np_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.bool:
                    np_inputs[k] = v.contiguous().numpy()
                elif v.dtype == torch.long or v.dtype == torch.int32:
                    np_inputs[k] = v.contiguous().numpy()
                else:
                    np_inputs[k] = v.to(torch.float32).contiguous().numpy()
            else:
                np_inputs[k] = v
        result = compiled(np_inputs)
        outputs = {}
        for i, out in enumerate(compiled.outputs):
            name = next(iter(out.get_names()))
            outputs[name] = torch.from_numpy(result[out])
        return outputs

    @staticmethod
    def _create_coordinate_grid(size, start=None, device=None):
        """Create a coordinate grid. Same as ZImageTransformer2DModel.create_coordinate_grid."""
        if start is None:
            start = (0,) * len(size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def _patchify_image(self, image, patch_size=2, f_patch_size=1):
        """Patchify an image tensor: [C, F, H, W] -> [num_patches, patch_dim]."""
        pH = pW = patch_size
        pF = f_patch_size
        C, F, H, W = image.size()
        F_t, H_t, W_t = F // pF, H // pH, W // pW
        image = image.view(C, F_t, pF, H_t, pH, W_t, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_t * H_t * W_t, pF * pH * pW * C)
        return image, (F, H, W), (F_t, H_t, W_t)

    def _pad_with_ids(self, feat, pos_grid_size, pos_start, device, noise_mask_val=None):
        """Pad feature to SEQ_MULTI_OF, create position IDs and pad mask. Matches original exactly."""
        ori_len = len(feat)
        pad_len = (-ori_len) % self.SEQ_MULTI_OF
        total_len = ori_len + pad_len

        ori_pos_ids = self._create_coordinate_grid(pos_grid_size, pos_start, device).flatten(0, 2)
        if pad_len > 0:
            pad_pos = self._create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(pad_len, 1)
            pos_ids = torch.cat([ori_pos_ids, pad_pos], dim=0)
            padded_feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
            pad_mask = torch.cat([
                torch.zeros(ori_len, dtype=torch.bool, device=device),
                torch.ones(pad_len, dtype=torch.bool, device=device),
            ])
        else:
            pos_ids = ori_pos_ids
            padded_feat = feat
            pad_mask = torch.zeros(ori_len, dtype=torch.bool, device=device)

        noise_mask = [noise_mask_val] * total_len if noise_mask_val is not None else None
        return padded_feat, pos_ids, pad_mask, total_len, noise_mask

    def _patchify_and_embed_omni(
        self, all_x, all_cap_feats, all_siglip_feats, images_noise_mask, patch_size=2, f_patch_size=1
    ):
        """
        Python-side patchification for omni mode. Matches original exactly.
        all_x: list[list[Tensor]] - [[cond_lat, target_lat], ...]
        all_cap_feats: list[list[Tensor]] - [[cond_cap, target_cap], ...]
        all_siglip_feats: list[list[Tensor|None]] - [[cond_siglip, None], ...]
        images_noise_mask: list[list[int]] - [[0, 1], ...]
        """
        bsz = len(all_x)
        device = all_x[0][-1].device
        dtype = all_x[0][-1].dtype

        all_x_out, all_x_size, all_x_pos, all_x_mask, all_x_len, all_x_noise = [], [], [], [], [], []
        all_cap_out, all_cap_pos, all_cap_mask, all_cap_len, all_cap_noise = [], [], [], [], []
        all_sig_out, all_sig_pos, all_sig_mask, all_sig_len, all_sig_noise = [], [], [], [], []

        for i in range(bsz):
            num_images = len(all_x[i])
            cap_feats_list, cap_pos_list, cap_mask_list, cap_lens, cap_noise = [], [], [], [], []
            cap_end_pos = []
            cap_cu_len = 1

            # Process captions
            for j, cap_item in enumerate(all_cap_feats[i]):
                noise_val = images_noise_mask[i][j] if j < len(images_noise_mask[i]) else 1
                cap_out, cap_pos_j, cap_mask_j, cap_len, cap_nm = self._pad_with_ids(
                    cap_item,
                    (len(cap_item) + (-len(cap_item)) % self.SEQ_MULTI_OF, 1, 1),
                    (cap_cu_len, 0, 0), device, noise_val,
                )
                cap_feats_list.append(cap_out)
                cap_pos_list.append(cap_pos_j)
                cap_mask_list.append(cap_mask_j)
                cap_lens.append(cap_len)
                cap_noise.extend(cap_nm)
                cap_cu_len += len(cap_item)
                cap_end_pos.append(cap_cu_len)
                cap_cu_len += 2  # for image vae and siglip tokens

            all_cap_out.append(torch.cat(cap_feats_list, dim=0))
            all_cap_pos.append(torch.cat(cap_pos_list, dim=0))
            all_cap_mask.append(torch.cat(cap_mask_list, dim=0))
            all_cap_len.append(cap_lens)
            all_cap_noise.append(cap_noise)

            # Process images
            x_feats_list, x_pos_list, x_mask_list, x_lens, x_size, x_noise = [], [], [], [], [], []
            siglip_feat_dim = self._transformer_config.get("siglip_feat_dim", 1152)
            for j, x_item in enumerate(all_x[i]):
                noise_val = images_noise_mask[i][j]
                if x_item is not None:
                    x_patches, size, (F_t, H_t, W_t) = self._patchify_image(x_item, patch_size, f_patch_size)
                    x_out, x_pos_j, x_mask_j, x_len, x_nm = self._pad_with_ids(
                        x_patches, (F_t, H_t, W_t), (cap_end_pos[j], 0, 0), device, noise_val
                    )
                    x_size.append(size)
                else:
                    x_len = self.SEQ_MULTI_OF
                    x_out = torch.zeros((x_len, 64), dtype=dtype, device=device)
                    x_pos_j = self._create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(x_len, 1)
                    x_mask_j = torch.ones(x_len, dtype=torch.bool, device=device)
                    x_nm = [noise_val] * x_len
                    x_size.append(None)
                x_feats_list.append(x_out)
                x_pos_list.append(x_pos_j)
                x_mask_list.append(x_mask_j)
                x_lens.append(x_len)
                x_noise.extend(x_nm)

            all_x_out.append(torch.cat(x_feats_list, dim=0))
            all_x_pos.append(torch.cat(x_pos_list, dim=0))
            all_x_mask.append(torch.cat(x_mask_list, dim=0))
            all_x_size.append(x_size)
            all_x_len.append(x_lens)
            all_x_noise.append(x_noise)

            # Process siglip
            if all_siglip_feats[i] is None:
                all_sig_len.append([0] * num_images)
                all_sig_out.append(None)
            else:
                sig_feats_list, sig_pos_list, sig_mask_list, sig_lens, sig_noise = [], [], [], [], []
                for j, sig_item in enumerate(all_siglip_feats[i]):
                    noise_val = images_noise_mask[i][j]
                    if sig_item is not None:
                        sig_H, sig_W, sig_C = sig_item.size()
                        sig_flat = sig_item.permute(2, 0, 1).reshape(sig_H * sig_W, sig_C)
                        sig_out, sig_pos_j, sig_mask_j, sig_len, sig_nm = self._pad_with_ids(
                            sig_flat, (1, sig_H, sig_W), (cap_end_pos[j] + 1, 0, 0), device, noise_val
                        )
                        # Scale position IDs to match image resolution
                        if x_size[j] is not None:
                            sig_pos_j = sig_pos_j.float()
                            sig_pos_j[..., 1] = sig_pos_j[..., 1] / max(sig_H - 1, 1) * (x_size[j][1] - 1)
                            sig_pos_j[..., 2] = sig_pos_j[..., 2] / max(sig_W - 1, 1) * (x_size[j][2] - 1)
                            sig_pos_j = sig_pos_j.to(torch.int32)
                    else:
                        sig_len = self.SEQ_MULTI_OF
                        sig_out = torch.zeros((sig_len, siglip_feat_dim), dtype=dtype, device=device)
                        sig_pos_j = self._create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(sig_len, 1)
                        sig_mask_j = torch.ones(sig_len, dtype=torch.bool, device=device)
                        sig_nm = [noise_val] * sig_len
                    sig_feats_list.append(sig_out)
                    sig_pos_list.append(sig_pos_j)
                    sig_mask_list.append(sig_mask_j)
                    sig_lens.append(sig_len)
                    sig_noise.extend(sig_nm)

                all_sig_out.append(torch.cat(sig_feats_list, dim=0))
                all_sig_pos.append(torch.cat(sig_pos_list, dim=0))
                all_sig_mask.append(torch.cat(sig_mask_list, dim=0))
                all_sig_len.append(sig_lens)
                all_sig_noise.append(sig_noise)

        # x position offsets (for unpatchify)
        all_x_pos_offsets = [(sum(all_cap_len[i]), sum(all_cap_len[i]) + sum(all_x_len[i])) for i in range(bsz)]

        return {
            "x_out": all_x_out, "x_pos": all_x_pos, "x_mask": all_x_mask,
            "x_len": all_x_len, "x_noise": all_x_noise, "x_size": all_x_size,
            "cap_out": all_cap_out, "cap_pos": all_cap_pos, "cap_mask": all_cap_mask,
            "cap_len": all_cap_len, "cap_noise": all_cap_noise,
            "sig_out": all_sig_out, "sig_pos": all_sig_pos, "sig_mask": all_sig_mask,
            "sig_len": all_sig_len, "sig_noise": all_sig_noise,
            "x_pos_offsets": all_x_pos_offsets,
        }

    def _run_split_transformer_step(self, x_combined, cap_feats, siglip_feats, image_noise_mask, timestep):
        """
        Run one denoising step through split transformer sub-models.
        Matches original ZImageTransformer2DModel.forward() logic exactly.

        Args:
            x_combined: list[list[Tensor]] - [[cond_lat, target_lat], ...] per batch
            cap_feats: list[list[Tensor]] - [[cond_cap, target_cap], ...] per batch
            siglip_feats: list[list[Tensor|None]] - [[siglip_3d, None], ...] per batch
            image_noise_mask: list[list[int]] - [[0, 1], ...] per batch
            timestep: Tensor [batch_size]

        Returns: list[Tensor] - output per batch item [C, F, H, W]
        """
        patch_size = 2
        f_patch_size = 1
        device = x_combined[0][-1].device

        # Step 1: Patchify (Python-side)
        data = self._patchify_and_embed_omni(
            x_combined, cap_feats, siglip_feats, image_noise_mask, patch_size, f_patch_size
        )

        bsz = len(x_combined)
        results = []

        # Process each batch item through the split models
        for bi in range(bsz):
            x_raw = data["x_out"][bi]         # [x_total_len, patch_dim]
            x_mask = data["x_mask"][bi]       # [x_total_len]
            x_pos = data["x_pos"][bi]         # [x_total_len, 3]
            x_noise_list = data["x_noise"][bi]  # list[int]

            cap_raw = data["cap_out"][bi]     # [cap_total_len, cap_dim]
            cap_mask = data["cap_mask"][bi]   # [cap_total_len]
            cap_pos = data["cap_pos"][bi]     # [cap_total_len, 3]
            cap_noise_list = data["cap_noise"][bi]

            # Step 2: Embed patches + timestep (OV sub-model)
            if data["sig_out"][bi] is not None:
                sig_raw = data["sig_out"][bi]
                sig_mask = data["sig_mask"][bi]
                sig_pos = data["sig_pos"][bi]
                sig_noise_list = data["sig_noise"][bi]
            else:
                siglip_feat_dim = self._transformer_config.get("siglip_feat_dim", 1152)
                sig_raw = torch.zeros(self.SEQ_MULTI_OF, siglip_feat_dim, device=device)
                sig_mask = torch.ones(self.SEQ_MULTI_OF, dtype=torch.bool, device=device)
                sig_pos = self._create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(self.SEQ_MULTI_OF, 1)
                sig_noise_list = [0] * self.SEQ_MULTI_OF

            embed_out = self._call_split_model("transformer_patch_embed", {
                "x_patches": x_raw,
                "x_pad_mask": x_mask,
                "cap_feats": cap_raw,
                "cap_pad_mask": cap_mask,
                "sig_feats": sig_raw,
                "sig_pad_mask": sig_mask,
                "timestep": timestep[bi:bi+1],
            })

            x_emb = embed_out["x_emb"]      # [x_total_len, dim]
            cap_emb = embed_out["cap_emb"]   # [cap_total_len, dim]
            sig_emb = embed_out["sig_emb"]   # [sig_total_len, dim]
            t_noisy = embed_out["t_noisy"]   # [1, adaln_dim]
            t_clean = embed_out["t_clean"]   # [1, adaln_dim]

            # Step 3: Compute RoPE embeddings (Python-side)
            # Note: pos_ids may be longer than features (original uses trim after pad_sequence)
            x_freqs = self._rope_embed(x_pos)[:x_emb.shape[0]]
            cap_freqs = self._rope_embed(cap_pos)[:cap_emb.shape[0]]
            sig_freqs = self._rope_embed(sig_pos)[:sig_emb.shape[0]]

            # Create attention masks (True = valid)
            x_attn = (~x_mask).unsqueeze(0)        # [1, x_total_len]
            cap_attn = (~cap_mask).unsqueeze(0)
            sig_attn = (~sig_mask).unsqueeze(0)

            # Create noise masks
            x_noise_mask = torch.tensor(x_noise_list, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq]
            cap_noise_mask_t = torch.tensor(cap_noise_list, dtype=torch.long, device=device)
            sig_noise_mask_t = torch.tensor(sig_noise_list, dtype=torch.long, device=device)

            # Step 4: Noise refiner (2 blocks with dual modulation)
            nr_out = self._call_split_model("transformer_noise_refiner", {
                "x": x_emb.unsqueeze(0),
                "attn_mask": x_attn,
                "freqs_cis": x_freqs.unsqueeze(0),
                "noise_mask": x_noise_mask,
                "adaln_noisy": t_noisy,
                "adaln_clean": t_clean,
            })
            x_refined = nr_out["output"]  # [1, x_total_len, dim]

            # Step 5: Context refiner (2 blocks, no modulation)
            cr_out = self._call_split_model("transformer_context_refiner", {
                "x": cap_emb.unsqueeze(0),
                "attn_mask": cap_attn,
                "freqs_cis": cap_freqs.unsqueeze(0),
            })
            cap_refined = cr_out["output"]  # [1, cap_total_len, dim]

            # Step 6: SigLIP refiner (2 blocks, no modulation)
            sr_out = self._call_split_model("transformer_siglip_refiner", {
                "x": sig_emb.unsqueeze(0),
                "attn_mask": sig_attn,
                "freqs_cis": sig_freqs.unsqueeze(0),
            })
            sig_refined = sr_out["output"]  # [1, sig_total_len, dim]

            # Step 7: Build unified sequence [cap, x, siglip] (Python-side)
            x_len = x_emb.shape[0]
            cap_len = cap_emb.shape[0]
            sig_len = sig_emb.shape[0]

            unified = torch.cat([cap_refined[0, :cap_len], x_refined[0, :x_len], sig_refined[0, :sig_len]], dim=0)
            unified_freqs = torch.cat([cap_freqs[:cap_len], x_freqs[:x_len], sig_freqs[:sig_len]], dim=0)
            unified_noise = torch.tensor(
                cap_noise_list + x_noise_list + sig_noise_list, dtype=torch.long, device=device
            )
            unified_attn = torch.cat([cap_attn[0], x_attn[0], sig_attn[0]], dim=0)

            # Step 8: Main transformer layers + final layer (OV sub-model)
            mt_out = self._call_split_model("transformer_main", {
                "x": unified.unsqueeze(0),
                "attn_mask": unified_attn.unsqueeze(0),
                "freqs_cis": unified_freqs.unsqueeze(0),
                "noise_mask": unified_noise.unsqueeze(0),
                "adaln_noisy": t_noisy,
                "adaln_clean": t_clean,
            })
            unified_out = mt_out["output"][0]  # [total_seq, out_dim]

            # Step 9: Unpatchify (Python-side)
            x_pos_offsets = data["x_pos_offsets"][bi]
            x_sizes = data["x_size"][bi]  # list of (F, H, W) or None
            out_channels = self._transformer_config.get("in_channels", 16)

            # Extract the x section from unified output
            x_section = unified_out[x_pos_offsets[0]:x_pos_offsets[1]]

            # Walk through images to find the target (last image)
            cu_len = 0
            x_item = None
            for j in range(len(x_sizes)):
                if x_sizes[j] is None:
                    cu_len += self.SEQ_MULTI_OF
                else:
                    F, H, W = x_sizes[j]
                    ori_len = (F // f_patch_size) * (H // patch_size) * (W // patch_size)
                    pad_len = (-ori_len) % self.SEQ_MULTI_OF
                    x_item = (
                        x_section[cu_len:cu_len + ori_len]
                        .view(F // f_patch_size, H // patch_size, W // patch_size,
                              f_patch_size, patch_size, patch_size, out_channels)
                        .permute(6, 0, 3, 1, 4, 2, 5)
                        .reshape(out_channels, F, H, W)
                    )
                    cu_len += ori_len + pad_len

            results.append(x_item)  # Only the last (target) image

        return results

    def _encode_prompt(self, prompt, device=None, prompt_embeds=None, max_sequence_length=512, num_condition_images=0):
        """Encode prompt for Omni mode. Returns list of list[Tensor] per batch item."""
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            if num_condition_images == 0:
                prompt[i] = ["<|im_start|>user\n" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n"]
            elif num_condition_images > 0:
                prompt_list = ["<|im_start|>user\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|vision_start|>"] * (num_condition_images - 1)
                prompt_list += ["<|vision_end|>" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|im_end|>"]
                prompt[i] = prompt_list

        flattened_prompt = []
        prompt_list_lengths = []
        for i in range(len(prompt)):
            prompt_list_lengths.append(len(prompt[i]))
            flattened_prompt.extend(prompt[i])

        text_inputs = self.tokenizer(
            flattened_prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        te_output = self.text_encoder(input_ids=text_input_ids, attention_mask=prompt_masks)
        all_embeds = te_output.last_hidden_state

        embeddings_list = []
        start_idx = 0
        for i in range(len(prompt_list_lengths)):
            batch_embeddings = []
            end_idx = start_idx + prompt_list_lengths[i]
            for j in range(start_idx, end_idx):
                batch_embeddings.append(all_embeds[j][prompt_masks[j]])
            embeddings_list.append(batch_embeddings)
            start_idx = end_idx

        return embeddings_list

    def _call_ov_siglip(self, siglip_inputs):
        """Call OV siglip model."""
        self.siglip.compile()
        model_inputs = {}
        for key in ["pixel_values", "pixel_attention_mask", "spatial_shapes"]:
            if hasattr(siglip_inputs, key):
                val = getattr(siglip_inputs, key)
                if isinstance(val, torch.Tensor):
                    model_inputs[key] = val.to(torch.float32) if val.is_floating_point() else val
                else:
                    model_inputs[key] = val
            elif isinstance(siglip_inputs, dict) and key in siglip_inputs:
                val = siglip_inputs[key]
                if isinstance(val, torch.Tensor):
                    model_inputs[key] = val.to(torch.float32) if val.is_floating_point() else val
                else:
                    model_inputs[key] = val
        ov_outputs = self.siglip.request(model_inputs, share_inputs=True).to_dict()
        return torch.from_numpy(next(iter(ov_outputs.values())))

    @torch.no_grad()
    def __call__(
        self,
        image=None,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps=50,
        sigmas=None,
        guidance_scale=5.0,
        cfg_normalization=False,
        cfg_truncation=1.0,
        negative_prompt=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        joint_attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    ):
        from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, retrieve_timesteps
        from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput

        if image is not None and not isinstance(image, list):
            image = [image]
        num_condition_images = len(image) if image is not None else 0

        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        # Encode prompts
        if prompt_embeds is None or prompt is not None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
                num_condition_images=num_condition_images,
            )

        # Process condition images
        condition_images = []
        resized_images = []
        if image is not None:
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    if height is not None and width is not None:
                        img = self.image_processor._resize_to_target_area(img, height * width)
                    else:
                        img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size
                resized_images.append(img)

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)

            if len(condition_images) > 0:
                height = height or image_height
                width = width or image_width
        else:
            height = height or 1024
            width = width or 1024

        num_channels_latents = self._transformer_config.get("in_channels", 16)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents,
            height, width, torch.float32, device, generator, latents,
        )

        # Encode condition images to latents via VAE encoder
        condition_latents = []
        if num_condition_images > 0 and self.vae_encoder is not None:
            for cimg in condition_images:
                cimg = cimg.to(device=device, dtype=torch.float32)
                vae_output = self.vae.encode(cimg)
                if hasattr(vae_output, 'latent_dist'):
                    clatent = vae_output.latent_dist.mode()[0]
                else:
                    clatent = vae_output[0]
                clatent = (clatent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                clatent = clatent.unsqueeze(1)  # [C, 1, H, W]
                condition_latents.append(clatent)

        # Replicate condition_latents for batch
        condition_latents_batch = [condition_latents.copy() for _ in range(batch_size * num_images_per_prompt)]
        if self.do_classifier_free_guidance:
            neg_condition_latents_batch = [[lat.clone() for lat in batch] for batch in condition_latents_batch]

        # Get SigLIP embeddings
        condition_siglip_embeds = []
        if num_condition_images > 0:
            for rimg in resized_images:
                siglip_inputs = self.siglip_processor(images=[rimg], return_tensors="pt").to(device)
                shape = siglip_inputs.spatial_shapes[0]
                hidden_state = self._call_ov_siglip(siglip_inputs)
                B, N, C = hidden_state.shape
                hidden_state = hidden_state[:, :shape[0] * shape[1]]
                hidden_state = hidden_state.view(shape[0], shape[1], C)
                condition_siglip_embeds.append(hidden_state)

        condition_siglip_batch = [condition_siglip_embeds.copy() for _ in range(batch_size * num_images_per_prompt)]
        if self.do_classifier_free_guidance:
            neg_siglip_batch = [[se.clone() for se in batch] for batch in condition_siglip_batch]

        # Format siglip: add None for target, wrap empty as None
        condition_siglip_batch = [None if sels == [] else sels + [None] for sels in condition_siglip_batch]
        if self.do_classifier_free_guidance:
            neg_siglip_batch = [None if sels == [] else sels + [None] for sels in neg_siglip_batch]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000
                t_norm = timestep[0].item()

                current_guidance_scale = self.guidance_scale
                if (self.do_classifier_free_guidance and self._cfg_truncation is not None
                        and float(self._cfg_truncation) <= 1):
                    if t_norm > self._cfg_truncation:
                        current_guidance_scale = 0.0

                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

                # Build inputs matching original pipeline format
                if apply_cfg:
                    latent_model_input = latents.repeat(2, 1, 1, 1)
                    pe_input = prompt_embeds + negative_prompt_embeds
                    cl_input = condition_latents_batch + neg_condition_latents_batch
                    cs_input = condition_siglip_batch + neg_siglip_batch
                    ts_input = timestep.repeat(2)
                else:
                    latent_model_input = latents
                    pe_input = prompt_embeds
                    cl_input = condition_latents_batch
                    cs_input = condition_siglip_batch
                    ts_input = timestep

                latent_model_input = latent_model_input.unsqueeze(2)  # add frame dim
                latent_list = list(latent_model_input.unbind(dim=0))

                current_batch = len(latent_list)

                # Build x_combined and image_noise_mask matching original format
                x_combined = [cl_input[j] + [latent_list[j]] for j in range(current_batch)]
                image_noise_mask = [[0] * len(cl_input[j]) + [1] for j in range(current_batch)]

                # Call split transformer
                model_out_list = self._run_split_transformer_step(
                    x_combined, pe_input, cs_input, image_noise_mask, ts_input
                )

                if apply_cfg:
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]
                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()
                        pred = pos + current_guidance_scale * (pos - neg)
                        if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(pred)
                            max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)
                        noise_pred.append(pred)
                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([item.float() for item in model_out_list], dim=0)

                noise_pred = noise_pred.squeeze(2) if noise_pred.ndim > 4 else noise_pred
                noise_pred = -noise_pred

                latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ZImagePipelineOutput(images=image)


SUPPORTED_OV_PIPELINES = [
    OVStableDiffusionPipeline,
    OVStableDiffusionImg2ImgPipeline,
    OVStableDiffusionInpaintPipeline,
    OVStableDiffusionXLPipeline,
    OVStableDiffusionXLImg2ImgPipeline,
    OVStableDiffusionXLInpaintPipeline,
    OVLatentConsistencyModelPipeline,
    OVLatentConsistencyModelImg2ImgPipeline,
]


def _get_ov_class(pipeline_class_name: str, throw_error_if_not_exist: bool = True):
    for ov_pipeline_class in SUPPORTED_OV_PIPELINES:
        if (
            ov_pipeline_class.__name__ == pipeline_class_name
            or ov_pipeline_class.auto_model_class.__name__ == pipeline_class_name
        ):
            return ov_pipeline_class

    if throw_error_if_not_exist:
        raise ValueError(f"OVDiffusionPipeline can't find a pipeline linked to {pipeline_class_name}")


OV_TEXT2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", OVStableDiffusionPipeline),
        ("stable-diffusion-xl", OVStableDiffusionXLPipeline),
        ("latent-consistency", OVLatentConsistencyModelPipeline),
    ]
)

OV_IMAGE2IMAGE_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", OVStableDiffusionImg2ImgPipeline),
        ("stable-diffusion-xl", OVStableDiffusionXLImg2ImgPipeline),
        ("latent-consistency", OVLatentConsistencyModelImg2ImgPipeline),
    ]
)

OV_INPAINT_PIPELINES_MAPPING = OrderedDict(
    [
        ("stable-diffusion", OVStableDiffusionInpaintPipeline),
        ("stable-diffusion-xl", OVStableDiffusionXLInpaintPipeline),
    ]
)

OV_TEXT2VIDEO_PIPELINES_MAPPING = OrderedDict()

if is_diffusers_version(">=", "0.32"):
    OV_TEXT2VIDEO_PIPELINES_MAPPING["ltx-video"] = OVLTXPipeline
    SUPPORTED_OV_PIPELINES.append(OVLTXPipeline)

if is_diffusers_version(">=", "0.29.0"):
    SUPPORTED_OV_PIPELINES.extend(
        [
            OVStableDiffusion3Pipeline,
            OVStableDiffusion3Img2ImgPipeline,
        ]
    )

    OV_TEXT2IMAGE_PIPELINES_MAPPING["stable-diffusion-3"] = OVStableDiffusion3Pipeline
    OV_IMAGE2IMAGE_PIPELINES_MAPPING["stable-diffusion-3"] = OVStableDiffusion3Img2ImgPipeline

if is_diffusers_version(">=", "0.30.0"):
    SUPPORTED_OV_PIPELINES.extend([OVStableDiffusion3InpaintPipeline, OVFluxPipeline])
    OV_INPAINT_PIPELINES_MAPPING["stable-diffusion-3"] = OVStableDiffusion3InpaintPipeline
    OV_TEXT2IMAGE_PIPELINES_MAPPING["flux"] = OVFluxPipeline

if is_diffusers_version(">=", "0.31.0"):
    SUPPORTED_OV_PIPELINES.extend([OVFluxImg2ImgPipeline, OVFluxInpaintPipeline])
    OV_INPAINT_PIPELINES_MAPPING["flux"] = OVFluxInpaintPipeline
    OV_IMAGE2IMAGE_PIPELINES_MAPPING["flux"] = OVFluxImg2ImgPipeline

if is_diffusers_version(">=", "0.32.0"):
    OV_INPAINT_PIPELINES_MAPPING["flux-fill"] = OVFluxFillPipeline
    SUPPORTED_OV_PIPELINES.append(OVFluxFillPipeline)
    OV_TEXT2IMAGE_PIPELINES_MAPPING["sana"] = OVSanaPipeline
    SUPPORTED_OV_PIPELINES.append(OVSanaPipeline)


if is_diffusers_version(">=", "0.33.0"):
    SUPPORTED_OV_PIPELINES.append(OVSanaSprintPipeline)
    OV_TEXT2IMAGE_PIPELINES_MAPPING["sana-sprint"] = OVSanaSprintPipeline

if ZImagePipeline is not object:
    SUPPORTED_OV_PIPELINES.append(OVZImagePipeline)
    OV_TEXT2IMAGE_PIPELINES_MAPPING["z-image"] = OVZImagePipeline

if ZImageOmniPipeline is not object:
    SUPPORTED_OV_PIPELINES.append(OVZImageOmniPipeline)
    OV_TEXT2IMAGE_PIPELINES_MAPPING["z-image-omni"] = OVZImageOmniPipeline

SUPPORTED_OV_PIPELINES_MAPPINGS = [
    OV_TEXT2IMAGE_PIPELINES_MAPPING,
    OV_IMAGE2IMAGE_PIPELINES_MAPPING,
    OV_INPAINT_PIPELINES_MAPPING,
    OV_TEXT2VIDEO_PIPELINES_MAPPING,
]


def _get_task_ov_class(mapping, pipeline_class_name):
    def _get_model_name(pipeline_class_name):
        for ov_pipelines_mapping in SUPPORTED_OV_PIPELINES_MAPPINGS:
            for model_name, ov_pipeline_class in ov_pipelines_mapping.items():
                if (
                    ov_pipeline_class.__name__ == pipeline_class_name
                    or ov_pipeline_class.auto_model_class.__name__ == pipeline_class_name
                ):
                    return model_name

    model_name = _get_model_name(pipeline_class_name)

    if model_name is not None:
        task_class = mapping.get(model_name, None)
        if task_class is not None:
            return task_class

    raise ValueError(f"OVPipelineForTask can't find a pipeline linked to {pipeline_class_name} for {model_name}")


class OVPipelineForTask(ConfigMixin):
    auto_model_class = DiffusionPipeline
    config_name = "model_index.json"

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_or_path, **kwargs):
        load_config_kwargs = {
            "force_download": kwargs.get("force_download", False),
            "resume_download": kwargs.get("resume_download", None),
            "local_files_only": kwargs.get("local_files_only", False),
            "cache_dir": kwargs.get("cache_dir", None),
            "revision": kwargs.get("revision", None),
            "proxies": kwargs.get("proxies", None),
            "token": kwargs.get("token", None),
        }
        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        config = config[0] if isinstance(config, tuple) else config
        class_name = config["_class_name"]

        ov_pipeline_class = _get_task_ov_class(cls.ov_pipelines_mapping, class_name)

        return ov_pipeline_class.from_pretrained(pretrained_model_or_path, **kwargs)


class OVPipelineForText2Image(OVPipelineForTask):
    auto_model_class = AutoPipelineForText2Image
    ov_pipelines_mapping = OV_TEXT2IMAGE_PIPELINES_MAPPING
    export_feature = "text-to-image"


class OVPipelineForImage2Image(OVPipelineForTask):
    auto_model_class = AutoPipelineForImage2Image
    ov_pipelines_mapping = OV_IMAGE2IMAGE_PIPELINES_MAPPING
    export_feature = "image-to-image"


class OVPipelineForInpainting(OVPipelineForTask):
    auto_model_class = AutoPipelineForInpainting
    ov_pipelines_mapping = OV_INPAINT_PIPELINES_MAPPING
    export_feature = "inpainting"


class OVPipelineForText2Video(OVPipelineForTask):
    auto_model_class = DiffusionPipeline
    ov_pipelines_mapping = OV_TEXT2VIDEO_PIPELINES_MAPPING
    export_feature = "text-to-video"
