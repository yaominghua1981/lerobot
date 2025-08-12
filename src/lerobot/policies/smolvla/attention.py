import os
import torch
from torch import nn


class AttentionBackend:
    def __init__(self, dbg_print=print, num_attention_heads: int = 0, num_key_value_heads: int = 0):
        self._dbg_print = dbg_print
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

    def get(self):
        use_custom = bool(int(os.getenv("SMOLVLA_USE_CUSTOM_ATTENTION", "0")))
        if use_custom:
            self._dbg_print("🔬 get_attention_interface: 使用自定义 attention (回退)")
            return self.custom
        self._dbg_print("🔬 get_attention_interface: 使用优化的 attention (SDPA/cuBLAS)")
        return self.optimized

    def custom(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        dbg = self._dbg_print
        dbg("🚀 使用自定义高效attention计算...")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，禁止在 CPU 上运行 attention")
        if not (query_states.is_cuda and key_states.is_cuda and value_states.is_cuda):
            raise RuntimeError(
                f"Attention 张量必须在 CUDA 上: Q:{query_states.device}, K:{key_states.device}, V:{value_states.device}"
            )

        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        except Exception:
            pass

        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        scale = head_dim ** -0.5
        batch_size = query_states.shape[0]
        query_states = query_states.transpose(1, 2)  # [B,H,Lq,D]

        # 查询与键的双重分块
        q_chunk_size = 16
        k_block_size = 64
        bsize, num_heads_total, q_len, d = query_states.shape
        k_len = key_states.shape[2]

        att_output = torch.empty((bsize, num_heads_total, q_len, d), device=query_states.device, dtype=value_states.dtype)
        neg_inf = torch.finfo(torch.float32).min

        for qi in range(0, q_len, q_chunk_size):
            qj = min(qi + q_chunk_size, q_len)
            q_chunk = query_states[:, :, qi:qj, :].to(torch.float32)
            m = torch.full((bsize, num_heads_total, q_chunk.shape[2]), neg_inf, device=q_chunk.device)
            for kk in range(0, k_len, k_block_size):
                kl = min(kk + k_block_size, k_len)
                k_block = key_states[:, :, kk:kl, :].to(torch.float32)
                logits_block = (q_chunk.unsqueeze(3) * k_block.unsqueeze(2)).sum(dim=-1) * scale
                if attention_mask is not None:
                    mask_sub = attention_mask[:, qi:qj, kk:kl]
                    logits_block = torch.where(mask_sub.unsqueeze(1), logits_block, torch.tensor(neg_inf, device=logits_block.device))
                block_max = logits_block.max(dim=3).values
                m = torch.maximum(m, block_max)
                del k_block, logits_block, block_max
                torch.cuda.empty_cache()

            S = torch.zeros((bsize, num_heads_total, q_chunk.shape[2]), device=q_chunk.device, dtype=torch.float32)
            N = torch.zeros((bsize, num_heads_total, q_chunk.shape[2], d), device=q_chunk.device, dtype=torch.float32)
            for kk in range(0, k_len, k_block_size):
                kl = min(kk + k_block_size, k_len)
                k_block = key_states[:, :, kk:kl, :].to(torch.float32)
                v_block = value_states[:, :, kk:kl, :].to(torch.float32)
                logits_block = (q_chunk.unsqueeze(3) * k_block.unsqueeze(2)).sum(dim=-1) * scale
                if attention_mask is not None:
                    mask_sub = attention_mask[:, qi:qj, kk:kl]
                    logits_block = torch.where(mask_sub.unsqueeze(1), logits_block, torch.tensor(neg_inf, device=logits_block.device))
                exp_block = torch.exp(logits_block - m.unsqueeze(3))
                S = S + exp_block.sum(dim=3)
                weighted_v = (exp_block.unsqueeze(-1) * v_block.unsqueeze(2)).sum(dim=3)
                N = N + weighted_v
                del k_block, v_block, logits_block, exp_block, weighted_v
                torch.cuda.empty_cache()

            att_out_chunk = (N / (S.unsqueeze(-1) + 1e-8)).to(value_states.dtype)
            att_output[:, :, qi:qj, :] = att_out_chunk
            del q_chunk, m, S, N, att_out_chunk
            torch.cuda.empty_cache()

        att_output = att_output.transpose(1, 2)
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
        return att_output

    def optimized(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        dbg = self._dbg_print
        dbg("🚀 使用优化的attention计算...")

        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]
        key_states = key_states[:, :, :, None, :].expand(batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim)
        key_states = key_states.reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)
        value_states = value_states[:, :, :, None, :].expand(batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_mask = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            original_dtype = query_states.dtype
            if original_dtype == torch.float32:
                query_states = query_states.to(torch.bfloat16)
                key_states = key_states.to(torch.bfloat16)
                value_states = value_states.to(torch.bfloat16)
            att_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
            if original_dtype == torch.float32:
                att_output = att_output.to(original_dtype)
            dbg("✅ scaled_dot_product_attention 成功")
        except Exception:
            # 回退到eager实现
            att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            att_weights *= head_dim ** -0.5
            big_neg = torch.finfo(att_weights.dtype).min
            if attention_mask is not None:
                masked = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
            else:
                masked = att_weights
            probs = nn.functional.softmax(masked, dim=-1)
            att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.transpose(1, 2)
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
        return att_output


