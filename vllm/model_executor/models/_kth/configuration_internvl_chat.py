# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
from typing import Dict, Any, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class InternVLChatConfig(PretrainedConfig):
    model_type = 'internvl_chat'
    is_composition = True

    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        llm_model_name_or_path: Optional[str] = None,
        llm_trust_remote_code: bool = False,
        llm_use_pretrained: bool = False,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        ps_version="v1",
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        spatial_feature_mode="none",
        spatial_pool_size=None,
        spatial_crop_size=None,
        spatial_crop_stride=None,
        sfe_intermediate_size=512,
        sfe_scales=None,
        dfi_detail_kernel=16,
        dfi_detail_stride=4,
        # v5: attention modules (only effective with llava_sp_* modes)
        use_spatial_attention=False,
        use_channel_attention=False,
        use_token_self_attention=False,
        use_cross_attention_fusion=False,
        cross_attention_fusion_heads=8,
        cross_attention_fusion_concat_kth=True,
        channel_attention_reduction=16,
        token_self_attention_heads=8,
        # v7: spatial gate type (env var GATE_TYPE 우선, 그다음 이 config)
        spatial_gate_type="scalar",
        # v8: register_only mode 토큰 수
        register_num_tokens=12,
        # v8: register init mode (zero | small_random | fixed_zero | fixed_random)
        register_init_mode="small_random",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'architectures': ['InternVisionModel']}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if llm_config is None:
            llm_config = {'architectures': ['Qwen2ForCausalLM']}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')
        assert 'architectures' in llm_config, "Should specify architecture in llm_config"

        if isinstance(llm_config, dict):
            if llm_model_name_or_path is None:
                llm_model_name_or_path = llm_config.get("model_name_or_path") or llm_config.get("_name_or_path")
            if "trust_remote_code" in llm_config:
                llm_trust_remote_code = bool(llm_config["trust_remote_code"])

        if isinstance(vision_config, dict):
            self.vision_config = InternVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if llm_use_pretrained:
            if not llm_model_name_or_path:
                raise ValueError("llm_use_pretrained=True requires llm_model_name_or_path or llm_config._name_or_path")
            from transformers import AutoConfig
            self.llm_config = AutoConfig.from_pretrained(
                llm_model_name_or_path,
                trust_remote_code=llm_trust_remote_code,
            )
            if not getattr(self.llm_config, "architectures", None):
                self.llm_config.architectures = ["AutoModelForCausalLM"]
            # HyperCLOVA models may store rope_scaling/rope_type as None or "default".
            # Force linear scaling with factor=1.0 to avoid ROPE_INIT_FUNCTIONS["default"] issues.
            rope_scaling = getattr(self.llm_config, "rope_scaling", None)
            rope_scaling_type = None
            if isinstance(rope_scaling, dict):
                rope_scaling_type = rope_scaling.get("rope_type")
            elif rope_scaling is not None:
                rope_scaling_type = getattr(rope_scaling, "rope_type", None)
            if rope_scaling is None or rope_scaling_type == "default":
                self.llm_config.rope_scaling = {"rope_type": "linear", "factor": 1.0}
        elif isinstance(llm_config, dict):
            architecture: str = llm_config['architectures'][0]
            if architecture == 'LlamaForCausalLM':
                from transformers import LlamaConfig
                self.llm_config = LlamaConfig(**llm_config)
            elif architecture == 'Qwen2ForCausalLM':
                from transformers import Qwen2Config
                self.llm_config = Qwen2Config(**llm_config)
            elif architecture == 'Qwen3MoeForCausalLM':
                from transformers import Qwen3MoeConfig
                self.llm_config = Qwen3MoeConfig(**llm_config)
            elif architecture == 'Qwen3ForCausalLM':
                from transformers import Qwen3Config
                self.llm_config = Qwen3Config(**llm_config)
            else:
                raise ValueError('Unsupported architecture: {}'.format(architecture))
        else:
            self.llm_config = llm_config

        self.llm_use_pretrained = llm_use_pretrained
        self.llm_model_name_or_path = llm_model_name_or_path
        self.llm_trust_remote_code = llm_trust_remote_code
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.spatial_feature_mode = spatial_feature_mode
        self.spatial_pool_size = spatial_pool_size
        self.spatial_crop_size = spatial_crop_size
        self.spatial_crop_stride = spatial_crop_stride
        self.sfe_intermediate_size = sfe_intermediate_size
        self.sfe_scales = sfe_scales
        self.dfi_detail_kernel = dfi_detail_kernel
        self.dfi_detail_stride = dfi_detail_stride
        self.use_spatial_attention = use_spatial_attention
        self.use_channel_attention = use_channel_attention
        self.use_token_self_attention = use_token_self_attention
        self.use_cross_attention_fusion = use_cross_attention_fusion
        self.cross_attention_fusion_heads = cross_attention_fusion_heads
        self.cross_attention_fusion_concat_kth = cross_attention_fusion_concat_kth
        self.channel_attention_reduction = channel_attention_reduction
        self.token_self_attention_heads = token_self_attention_heads
        self.spatial_gate_type = spatial_gate_type
        self.register_num_tokens = register_num_tokens
        self.register_init_mode = register_init_mode
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')
        logger.info(f'spatial_feature_mode: {self.spatial_feature_mode}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch
        output['spatial_feature_mode'] = self.spatial_feature_mode
        output['spatial_pool_size'] = self.spatial_pool_size
        output['spatial_crop_size'] = self.spatial_crop_size
        output['spatial_crop_stride'] = self.spatial_crop_stride
        output['sfe_intermediate_size'] = self.sfe_intermediate_size
        output['sfe_scales'] = self.sfe_scales
        output['dfi_detail_kernel'] = self.dfi_detail_kernel
        output['dfi_detail_stride'] = self.dfi_detail_stride
        output['use_spatial_attention'] = self.use_spatial_attention
        output['use_channel_attention'] = self.use_channel_attention
        output['use_token_self_attention'] = self.use_token_self_attention
        output['channel_attention_reduction'] = self.channel_attention_reduction
        output['token_self_attention_heads'] = self.token_self_attention_heads
        output['spatial_gate_type'] = self.spatial_gate_type
        output['register_num_tokens'] = self.register_num_tokens

        return output
