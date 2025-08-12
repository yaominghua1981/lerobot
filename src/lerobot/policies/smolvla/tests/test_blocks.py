import os
import torch
from torch import nn

from ..blocks import CrossAttnBlocks


class DummyLayer(nn.Module):
    def __init__(self, hidden_size=32, heads=4, head_dim=8):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, heads * head_dim)
        self.self_attn.k_proj = nn.Linear(hidden_size, heads * head_dim)
        self.self_attn.v_proj = nn.Linear(hidden_size, heads * head_dim)
        self.self_attn.o_proj = nn.Linear(heads * head_dim, hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.head_dim = head_dim


class DummyModel:
    def __init__(self, num_layers=2, hidden_size=32, heads=4, head_dim=8):
        self.layers = nn.ModuleList([DummyLayer(hidden_size, heads, head_dim) for _ in range(num_layers)])


def build_model_layers(num_vlm_layers=2):
    vlm = DummyModel(num_layers=num_vlm_layers)
    expert = DummyModel(num_layers=num_vlm_layers)
    return [vlm.layers, expert.layers]


def test_prefix_kv_fill_and_reuse():
    torch.manual_seed(0)
    B, Lp, Ls, H = 2, 4, 3, 32
    heads, head_dim = 4, 8

    dbg = lambda *a, **k: None
    get_layers = lambda models: models  # unused in test
    blocks = CrossAttnBlocks(dbg, get_layers, num_vlm_layers=2, num_attention_heads=heads, num_key_value_heads=heads)

    model_layers = build_model_layers(num_vlm_layers=2)
    prefix = torch.randn(B, Lp, H)
    suffix = torch.randn(B, Ls, H)
    pos = torch.arange(Lp + Ls).unsqueeze(0).repeat(B, 1)
    attn_mask = torch.ones(B, Lp + Ls, Lp + Ls, dtype=torch.bool)

    # Step 1: fill kv cache with prefix only
    outputs, past = blocks.forward_cross_attn_layer(
        model_layers,
        [prefix, None],
        layer_idx=0,
        position_ids=pos[:, :Lp],
        attention_mask=attn_mask[:, :Lp, :Lp],
        batch_size=B,
        head_dim=head_dim,
        use_cache=True,
        fill_kv_cache=True,
        past_key_values=None,
    )
    assert isinstance(past, dict) and 0 in past and 'key_states' in past[0]

    # Step 2: reuse kv for suffix
    outputs2, _ = blocks.forward_cross_attn_layer(
        model_layers,
        [None, suffix],
        layer_idx=0,
        position_ids=pos[:, Lp:],
        attention_mask=attn_mask[:, Lp:, :],
        batch_size=B,
        head_dim=head_dim,
        use_cache=True,
        fill_kv_cache=False,
        past_key_values=past,
    )
    assert outputs2[1] is not None and outputs2[1].shape == (B, Ls, H)


