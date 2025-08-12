from typing import List, Optional
import os
import builtins
import torch
from torch import nn

from .attention import AttentionBackend
from .smolvlm_utils import apply_rope
from .cuda_safe_layers import CudaSafeLinear
from .ops import custom_mlp_forward, custom_linear_forward


class CrossAttnBlocks:
    def __init__(
        self,
        dbg_print,
        get_model_layers,
        num_vlm_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        custom_mlp_forward=None,
        custom_linear_forward=None,
    ):
        self._dbg_print = dbg_print
        self.get_model_layers = get_model_layers
        self.num_vlm_layers = num_vlm_layers
        self.attn = AttentionBackend(dbg_print, num_attention_heads, num_key_value_heads)
        self.custom_mlp_forward = custom_mlp_forward or custom_mlp_forward
        self.custom_linear_forward = custom_linear_forward or custom_linear_forward

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ):
        print = self._dbg_print
        print(f"🔬 forward_attn_layer: 第 {layer_idx} 层开始...")
        vlm_layers = model_layers[0]
        expert_layers = model_layers[1]
        att_outputs: List[Optional[torch.Tensor]] = []
        past_key_values_list: List[Optional[dict]] = []

        for i, hidden_states in enumerate(inputs_embeds):
            print(f"🔬 forward_attn_layer: 处理第 {i} 个embedding...")
            if hidden_states is None:
                att_outputs.append(None)
                past_key_values_list.append(None)
                continue

            layer = vlm_layers[layer_idx] if i == 0 else (expert_layers[layer_idx] if expert_layers[layer_idx] is not None else vlm_layers[layer_idx])
            if layer is None:
                att_outputs.append(None)
                past_key_values_list.append(None)
                continue

            # 统一按 4D 形状计算注意力：[B, L, Heads, HeadDim]
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, head_dim)

            if hasattr(layer, 'input_layernorm'):
                hidden_states = layer.input_layernorm(hidden_states)
            elif hasattr(layer, 'ln_1'):
                hidden_states = layer.ln_1(hidden_states)
            elif hasattr(layer, 'norm1'):
                hidden_states = layer.norm1(hidden_states)
            else:
                if hasattr(layer, 'ln_1'):
                    layer.input_layernorm = layer.ln_1
                    hidden_states = layer.input_layernorm(hidden_states)
                elif hasattr(layer, 'norm1'):
                    layer.input_layernorm = layer.norm1
                    hidden_states = layer.input_layernorm(hidden_states)
                else:
                    att_outputs.append(None)
                    past_key_values_list.append(None)
                    continue

            if not hasattr(layer, 'self_attn'):
                att_outputs.append(None)
                past_key_values_list.append(None)
                continue

            has_q = hasattr(layer.self_attn, 'q_proj')
            has_k = hasattr(layer.self_attn, 'k_proj')
            has_v = hasattr(layer.self_attn, 'v_proj')
            has_o = hasattr(layer.self_attn, 'o_proj')
            if not (has_q and has_k and has_v):
                att_outputs.append(None)
                past_key_values_list.append(None)
                continue

            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            attention_interface = self.attn.get()
            att_output = attention_interface(attention_mask, batch_size, head_dim, query_state, key_state, value_state)

            if has_o:
                att_output = layer.self_attn.o_proj(att_output)
            hidden_states = att_output + inputs_embeds[i]
            after_first_residual = hidden_states.clone()

            if hasattr(layer, 'post_attention_layernorm'):
                hidden_states = layer.post_attention_layernorm(hidden_states)
            elif hasattr(layer, 'ln_2'):
                hidden_states = layer.ln_2(hidden_states)
            elif hasattr(layer, 'norm2'):
                hidden_states = layer.norm2(hidden_states)

            if hasattr(layer, 'mlp'):
                if isinstance(layer.mlp, CudaSafeLinear):
                    hidden_states = layer.mlp(hidden_states)
                else:
                    hidden_states = (self.custom_mlp_forward or custom_mlp_forward)(hidden_states, layer.mlp)
            else:
                if hasattr(layer, 'linear1') and hasattr(layer, 'linear2'):
                    if isinstance(layer.linear1, CudaSafeLinear) and isinstance(layer.linear2, CudaSafeLinear):
                        hidden_states = layer.linear1(hidden_states)
                        hidden_states = layer.activation(hidden_states)
                        hidden_states = layer.dropout(hidden_states)
                        hidden_states = layer.linear2(hidden_states)
                        hidden_states = layer.dropout1(hidden_states)
                    else:
                        # 手动实现以避免 cuBLAS
                        if self.custom_linear_forward or custom_linear_forward:
                            hidden_states = (self.custom_linear_forward or custom_linear_forward)(hidden_states, layer.linear1)
                            hidden_states = layer.activation(hidden_states)
                            hidden_states = layer.dropout(hidden_states)
                            hidden_states = (self.custom_linear_forward or custom_linear_forward)(hidden_states, layer.linear2)
                            hidden_states = layer.dropout1(hidden_states)
                        else:
                            hidden_states = layer.linear1(hidden_states)
                            hidden_states = layer.activation(hidden_states)
                            hidden_states = layer.dropout(hidden_states)
                            hidden_states = layer.linear2(hidden_states)
                            hidden_states = layer.dropout1(hidden_states)

            hidden_states = hidden_states + after_first_residual
            att_outputs.append(hidden_states)

            if use_cache and fill_kv_cache:
                # Be robust to dict- or list-like past_key_values
                if isinstance(past_key_values, (list, tuple)):
                    if past_key_values is not None and len(past_key_values) > i and past_key_values[i] is not None:
                        past_key_values_list.append(past_key_values[i])
                    else:
                        past_key_values_list.append(None)
                elif isinstance(past_key_values, dict):
                    # Use layer index as key if available
                    past_key_values_list.append(past_key_values.get(layer_idx, None))
                else:
                    past_key_values_list.append(None)
            else:
                past_key_values_list.append(None)

        print(f"🔬 forward_attn_layer: 第 {layer_idx} 层完成")
        return att_outputs, past_key_values_list

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ):
        print = self._dbg_print
        print(f"🔬 forward_cross_attn_layer: 第 {layer_idx} 层开始...")
        attention_interface = self.attn.get()

        att_outputs: List[Optional[torch.Tensor]] = []
        past_key_values_list: List[Optional[dict]] = []
        # Initialize KV placeholders for prefix/cached reuse paths
        key_states: Optional[torch.Tensor] = None
        value_states: Optional[torch.Tensor] = None

        assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None and not fill_kv_cache), (
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        )

        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]

            if hasattr(layer, 'input_layernorm'):
                hidden_states = layer.input_layernorm(inputs_embeds[0])
            elif hasattr(layer, 'ln_1'):
                hidden_states = layer.ln_1(inputs_embeds[0])
            elif hasattr(layer, 'norm1'):
                hidden_states = layer.norm1(inputs_embeds[0])
            else:
                if hasattr(layer, 'ln_1'):
                    layer.input_layernorm = layer.ln_1
                    hidden_states = layer.input_layernorm(inputs_embeds[0])
                elif hasattr(layer, 'norm1'):
                    layer.input_layernorm = layer.norm1
                    hidden_states = layer.input_layernorm(inputs_embeds[0])
                else:
                    att_outputs.append(None)
                    past_key_values_list.append(None)
                    return att_outputs, past_key_values

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)
            # Ensure plural naming for downstream caching and expert projection
            value_states = value_state

            att_output = attention_interface(prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_state)
            att_output = layer.self_attn.o_proj(att_output)
            hidden_states = att_output + inputs_embeds[0]
            after_first_residual = hidden_states.clone()

            if hasattr(layer, 'post_attention_layernorm'):
                hidden_states = layer.post_attention_layernorm(hidden_states)
            elif hasattr(layer, 'ln_2'):
                hidden_states = layer.ln_2(hidden_states)
            elif hasattr(layer, 'norm2'):
                hidden_states = layer.norm2(hidden_states)
            else:
                if hasattr(layer, 'ln_2'):
                    layer.post_attention_layernorm = layer.ln_2
                    hidden_states = layer.post_attention_layernorm(hidden_states)
                elif hasattr(layer, 'norm2'):
                    layer.post_attention_layernorm = layer.norm2
                    hidden_states = layer.post_attention_layernorm(hidden_states)
                else:
                    att_outputs.append(None)
                    return att_outputs, past_key_values

            if hasattr(layer, 'mlp'):
                hidden_states = layer.mlp(hidden_states)
            else:
                hidden_states = layer.linear1(hidden_states)
                hidden_states = layer.activation(hidden_states)
                hidden_states = layer.dropout(hidden_states)
                hidden_states = layer.linear2(hidden_states)
                hidden_states = layer.dropout1(hidden_states)

            hidden_states = hidden_states + after_first_residual
            att_outputs.append(hidden_states)
            past_key_values_list.append(None)
        else:
            expert_position_id = position_ids
            att_outputs.append(None)
            past_key_values_list.append(None)

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                # Only write cache when KV were computed in prefix branch and storage is a dict
                if isinstance(past_key_values, dict) and key_states is not None and value_states is not None:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
            else:
                layer_cache = past_key_values.get(layer_idx) if isinstance(past_key_values, dict) else None
                if layer_cache is not None:
                    key_states = layer_cache.get("key_states", key_states)
                    value_states = layer_cache.get("value_states", value_states)

        # Expert
        expert_layer = model_layers[1][layer_idx]
        # Proceed only if KV are available (from prefix or cache)
        if expert_layer is not None and inputs_embeds[1] is not None and key_states is not None and value_states is not None:
            if hasattr(expert_layer, 'input_layernorm'):
                expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])
            elif hasattr(expert_layer, 'ln_1'):
                expert_hidden_states = expert_layer.ln_1(inputs_embeds[1])
            elif hasattr(expert_layer, 'norm1'):
                expert_hidden_states = expert_layer.norm1(inputs_embeds[1])
            else:
                if hasattr(expert_layer, 'ln_1'):
                    expert_layer.input_layernorm = expert_layer.ln_1
                    expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])
                elif hasattr(expert_layer, 'norm1'):
                    expert_layer.input_layernorm = expert_layer.norm1
                    expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])
                else:
                    att_outputs.append(None)
                    return att_outputs, past_key_values

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(*key_states.shape[:2], -1)
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(*_key_states.shape[:-1], -1, head_dim)

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(*value_states.shape[:2], -1)
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(*_value_states.shape[:-1], -1, head_dim)

            expert_position_id = (expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values)
            expert_attention_mask = attention_mask[:, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :]

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            expert_att_output = attention_interface(
                expert_attention_mask, batch_size, head_dim, expert_query_states, expert_key_states, expert_value_states
            )
            expert_att_output = expert_layer.self_attn.o_proj(expert_att_output)
            expert_hidden_states = expert_att_output + inputs_embeds[1]
            after_first_residual = expert_hidden_states.clone()

            if hasattr(expert_layer, 'post_attention_layernorm'):
                expert_hidden_states = expert_layer.post_attention_layernorm(expert_hidden_states)
            elif hasattr(expert_layer, 'ln_2'):
                expert_hidden_states = expert_layer.ln_2(expert_hidden_states)
            elif hasattr(expert_layer, 'norm2'):
                expert_hidden_states = expert_layer.norm2(expert_hidden_states)
            else:
                if hasattr(expert_layer, 'ln_2'):
                    expert_layer.post_attention_layernorm = expert_layer.ln_2
                    expert_hidden_states = expert_layer.post_attention_layernorm(expert_hidden_states)
                elif hasattr(expert_layer, 'norm2'):
                    expert_layer.post_attention_layernorm = expert_layer.norm2
                    expert_hidden_states = expert_layer.post_attention_layernorm(expert_hidden_states)
                else:
                    att_outputs.append(None)
                    return att_outputs, past_key_values

            if hasattr(expert_layer, 'mlp'):
                expert_hidden_states = expert_layer.mlp(expert_hidden_states)
            else:
                expert_hidden_states = expert_layer.linear1(expert_hidden_states)
                expert_hidden_states = expert_layer.activation(expert_hidden_states)
                expert_hidden_states = expert_layer.dropout(expert_hidden_states)
                expert_hidden_states = expert_layer.linear2(expert_hidden_states)
                expert_hidden_states = expert_layer.dropout1(expert_hidden_states)

            expert_hidden_states = expert_hidden_states + after_first_residual
            # One-line progress for expert branch (minimal overhead)
            try:
                if int(os.getenv("EVAL_VERBOSE", "0")) >= 1:
                    builtins.print(f"🔧 action expert: layer {layer_idx}, output={tuple(expert_hidden_states.shape)}")
            except Exception:
                pass
            att_outputs.append(expert_hidden_states)
        else:
            att_outputs.append(None)

        if use_cache and fill_kv_cache:
            if past_key_values is not None and len(past_key_values_list) < len(att_outputs):
                past_key_values_list.append(past_key_values.get(layer_idx, None))
            else:
                past_key_values_list.append(None)
        else:
            past_key_values_list.append(None)

        print(f"🔬 forward_cross_attn_layer: 第 {layer_idx} 层完成")
        return att_outputs, past_key_values



