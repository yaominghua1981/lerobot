import torch
from torch import nn


class CudaSafeLinear(nn.Module):
    """避免 cuBLAS 的线性层实现，使用逐元素乘与归约，保持 GPU 计算。"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.normal_(self.weight, 0.0, std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        try:
            if input.device != self.weight.device:
                input = input.to(self.weight.device)
            original_shape = input.shape
            if input.dim() == 1:
                input = input.unsqueeze(0)
                was_1d = True
            else:
                was_1d = False
            batch_dims = input.shape[:-1]
            batch_size = int(torch.prod(torch.tensor(batch_dims)))
            input_reshaped = input.reshape(batch_size, self.in_features)
            output = torch.zeros(batch_size, self.out_features, device=input.device, dtype=input.dtype)
            chunk_size = 32
            for i in range(0, self.out_features, chunk_size):
                end_i = min(i + chunk_size, self.out_features)
                weight_chunk = self.weight[i:end_i]
                for j in range(end_i - i):
                    weight_row = weight_chunk[j:j+1]
                    dot_product = (input_reshaped * weight_row).sum(dim=1)
                    output[:, i + j] = dot_product
                del weight_chunk
                torch.cuda.empty_cache()
            if self.bias is not None:
                output = output + self.bias
            if was_1d:
                output = output.squeeze(0)
            else:
                output = output.reshape(*batch_dims, self.out_features)
            return output
        except Exception as e:
            try:
                return nn.functional.linear(input, self.weight, self.bias)
            except Exception as cpu_e:
                raise RuntimeError(f"CudaSafeLinear 完全失败: {e}, CPU回退失败: {cpu_e}")


