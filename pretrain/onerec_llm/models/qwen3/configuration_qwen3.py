# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen3 model configuration"""

import math

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Qwen3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3Model`]. It is used to instantiate a
    Qwen3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-8B [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen3Model, Qwen3Config

    >>> # Initializing a Qwen3 style configuration
    >>> configuration = Qwen3Config()

    >>> # Initializing a model from the Qwen3-8B style configuration
    >>> model = Qwen3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        adapter_use_spm=True,
        adapter_use_itm=True,
        adapter_itm_order=1,
        adapter_itm_mode="default",
        adapter_itm_mapping=None,
        adapter_itm_prior_counts=None,
        adapter_itm_aux_loss_weight=0.0,
        adapter_itm_aux_head=False,
        adapter_itm_pattern_manifest=None,
        adapter_sid_begin_id=None,
        adapter_sid_end_id=None,
        adapter_sid_code_ranges=None,
        adapter_type=None,
        adapter_layers=None,
        houlsby_adapter=False,
        houlsby_adapter_bottleneck=256,
        adapter_hidden_size=None,
        adapter_spm_hidden_size=None,
        adapter_itm_hidden_size=None,
        adapter_memory_size=1024,
        adapter_spm_memory_size=None,
        adapter_itm_memory_size=None,
        adapter_memory_init="default",
        adapter_memory_init_std=0.01,
        adapter_memory_prototype_path=None,
        adapter_ngram_order=2,
        adapter_topk=4,
        adapter_use_alpha_layer=False,
        adapter_alpha_init=0.1,
        adapter_fixed_context_gate=None,
        adapter_use_conv=True,
        adapter_use_sid_span=False,
        adapter_sid_token_start=None,
        adapter_sid_token_end=None,
        adapter_lngram_bits_per_route=4,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window  # we check `use_sliding_window` in the modeling code
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.adapter_use_spm = adapter_use_spm
        self.adapter_use_itm = adapter_use_itm
        self.adapter_itm_order = adapter_itm_order
        self.adapter_itm_mode = adapter_itm_mode
        self.adapter_itm_prior_counts = adapter_itm_prior_counts
        try:
            adapter_itm_aux_loss_weight = float(adapter_itm_aux_loss_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("adapter_itm_aux_loss_weight must be finite and nonnegative") from exc
        if not math.isfinite(adapter_itm_aux_loss_weight) or adapter_itm_aux_loss_weight < 0:
            raise ValueError("adapter_itm_aux_loss_weight must be finite and nonnegative")
        if (adapter_itm_aux_loss_weight > 0 or adapter_itm_aux_head) and not (
            adapter_type == "engram_structure" and adapter_use_itm and adapter_itm_mode == "exact"
        ):
            raise ValueError("ITM auxiliary loss requires engram_structure with ITM enabled in exact mode")
        self.adapter_itm_aux_loss_weight = adapter_itm_aux_loss_weight
        self.adapter_itm_aux_head = bool(adapter_itm_aux_head or adapter_itm_aux_loss_weight > 0)
        self.adapter_itm_mapping = adapter_itm_mapping
        self.adapter_itm_pattern_manifest = adapter_itm_pattern_manifest
        self.adapter_sid_begin_id = adapter_sid_begin_id
        self.adapter_sid_end_id = adapter_sid_end_id
        self.adapter_sid_code_ranges = adapter_sid_code_ranges
        self.adapter_structure_version = 1
        self.adapter_type = adapter_type
        self.adapter_layers = adapter_layers
        self.houlsby_adapter = houlsby_adapter
        self.houlsby_adapter_bottleneck = houlsby_adapter_bottleneck
        self.adapter_hidden_size = adapter_hidden_size
        self.adapter_spm_hidden_size = adapter_spm_hidden_size
        self.adapter_itm_hidden_size = adapter_itm_hidden_size
        self.adapter_memory_size = adapter_memory_size
        self.adapter_spm_memory_size = adapter_spm_memory_size
        self.adapter_itm_memory_size = adapter_itm_memory_size
        self.adapter_memory_init = adapter_memory_init
        self.adapter_memory_init_std = adapter_memory_init_std
        self.adapter_memory_prototype_path = adapter_memory_prototype_path
        self.adapter_ngram_order = adapter_ngram_order
        self.adapter_topk = adapter_topk
        self.adapter_use_alpha_layer = adapter_use_alpha_layer
        self.adapter_alpha_init = adapter_alpha_init
        self.adapter_fixed_context_gate = adapter_fixed_context_gate
        self.adapter_use_conv = adapter_use_conv
        self.adapter_use_sid_span = adapter_use_sid_span
        self.adapter_sid_token_start = adapter_sid_token_start
        self.adapter_sid_token_end = adapter_sid_token_end
        self.adapter_lngram_bits_per_route = adapter_lngram_bits_per_route
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3Config"]
