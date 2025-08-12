import torch
from torch import nn


def custom_mlp_forward(hidden_states: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
    try:
        if hasattr(mlp, 'c_fc') and hasattr(mlp, 'c_proj') and hasattr(mlp, 'act'):
            x = custom_linear_forward(hidden_states, mlp.c_fc)
            x = mlp.act(x)
            x = custom_linear_forward(x, mlp.c_proj)
            return x
        return mlp(hidden_states)
    except Exception:
        return mlp(hidden_states)


def custom_linear_forward(hidden_states: torch.Tensor, linear_layer: nn.Linear) -> torch.Tensor:
    try:
        weight = linear_layer.weight
        bias = linear_layer.bias
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        original_shape = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, in_features)
        chunk_size = 32
        output = torch.zeros(hidden_states_reshaped.shape[0], out_features, device=hidden_states.device, dtype=hidden_states.dtype)
        for i in range(0, hidden_states_reshaped.shape[0], chunk_size):
            end_i = min(i + chunk_size, hidden_states_reshaped.shape[0])
            chunk = hidden_states_reshaped[i:end_i]
            for j in range(out_features):
                weight_row = weight[j:j+1]
                dot_product = (chunk * weight_row).sum(dim=1)
                output[i:end_i, j] = dot_product
            del chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if bias is not None:
            output = output + bias
        output = output.view(*original_shape[:-1], out_features)
        return output
    except Exception:
        return linear_layer(hidden_states)


