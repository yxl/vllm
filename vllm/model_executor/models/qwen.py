# coding=utf-8
# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
"""Inference-only QWen model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantized_linear import ParallelLinear
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.layers import VocabParallelEmbedding
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.quantization_utils import QuantizationConfig
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor,
    get_parallel_weight,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.qwen import QWenConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class QWenMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.gate_up_proj = ParallelLinear.column(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
        )
        self.c_proj = ParallelLinear.row(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class QWenAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # pylint: disable=invalid-name
        self.c_attn = ParallelLinear.column(
            hidden_size,
            3 * hidden_size,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
        )
        self.c_proj = ParallelLinear.row(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
        )
        self.scaling = self.head_dim**-0.5

        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        k_cache, v_cache = kv_cache
        attn_output = self.attn(
            positions, q, k, v, k_cache, v_cache, input_metadata, cache_event
        )

        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nn.Module):
    def __init__(
        self, config: QWenConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        rope_scaling = getattr(config, "rope_scaling", {})

        rope_theta = getattr(config, "rotary_emb_base", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        seq_length = getattr(config, "seq_length", 8192)

        if config.use_dynamic_ntk and config.use_logn_attn:
            rope_scaling["type"] = "dynamic-qwen"
            rope_scaling["seq_len"] = seq_length
            rope_scaling["factor"] = 2.1
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.rope_theta = rope_theta
        self.attn = QWenAttention(
            config.hidden_size,
            config.num_attention_heads,
            max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            quant_config=quant_config,
        )

        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = QWenMLP(
            config.hidden_size,
            config.intermediate_size // 2,
            quant_config=quant_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QWenModel(nn.Module):
    def __init__(
        self, config: QWenConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.wte = VocabParallelEmbedding(
            vocab_size,
            config.hidden_size,
        )
        self.h = nn.ModuleList(
            [QWenBlock(config, quant_config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.h)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.h[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class QWenLMHeadModel(nn.Module):
    def __init__(
        self, config: QWenConfig, quant_config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.transformer = QWenModel(config, quant_config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.lm_head = ParallelLinear.column(
            config.hidden_size,
            vocab_size,
            bias=False,
            gather_output=False,
            quant_config=None,
        )
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.transformer(
            input_ids, positions, kv_caches, input_metadata, cache_events
        )
        next_tokens = self.sampler(self.lm_head.weight, hidden_states, input_metadata)
        return next_tokens

    column_parallel_layers = []
    row_parallel_layers = ["c_proj"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        (
            column_parallel_weights,
            row_parallel_weights,
            ignore_weight_suffixes,
        ) = get_parallel_weight(self)
        tp_world_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            if any(name.endswith(suffix) for suffix in ignore_weight_suffixes):
                continue

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)

            is_transposed = False
            if self.quant_config is not None:
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = loaded_weight.T

            if "c_attn" in name and "g_idx" not in name:
                total_num_heads = self.config.num_attention_heads
                num_heads = total_num_heads // tp_world_size
                head_start = tp_rank * num_heads
                head_end = (tp_rank + 1) * num_heads

                weight_shape = loaded_weight.shape
                loaded_weight = loaded_weight.view(
                    3, total_num_heads, -1, *weight_shape[1:]
                )
                loaded_weight = loaded_weight[:, head_start:head_end]
                loaded_weight = loaded_weight.reshape(-1, *weight_shape[1:])

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["w2", "w1"]):
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, "gate_up_proj")
                if "g_idx" in name:
                    break
                if name not in state_dict:
                    continue
                param = state_dict[name]
                if is_transposed:
                    param = param.T
                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                param_slice = param.data[
                    shard_size * stride_id : shard_size * (stride_id + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            if name not in state_dict:
                continue
            param = state_dict[name]
            if is_transposed:
                param = param.T

            if "wte" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tp_rank,
            )
