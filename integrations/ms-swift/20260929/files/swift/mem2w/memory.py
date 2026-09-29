"""Persistent key/value memory used by the native ms-swift Mem2W path."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PersistentKVMemory(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.slots = config.slots
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.scale = config.key_dim**-0.5
        self.rms_eps = config.rms_eps
        self.compute_chunk_size = config.compute_chunk_size
        self.W_Q = nn.Parameter(torch.empty(config.hidden_size, config.key_dim))
        self.K = nn.Parameter(torch.empty(config.slots, config.key_dim))
        self.V = nn.Parameter(torch.empty(config.slots, config.value_dim))
        self.W_O = nn.Parameter(torch.empty(config.value_dim, config.hidden_size))
        for parameter in (self.W_Q, self.K, self.V, self.W_O):
            parameter.is_mem2w_parameter = True
        self.dropout = nn.Dropout(config.memory_dropout) if config.memory_dropout else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.normal_(self.K, std=1.0)
        nn.init.normal_(self.V, std=0.02)
        nn.init.zeros_(self.W_O)

    def forward(self, hidden_states: Tensor, *, enabled=True, attention_mask=None, stop_content_grad=False):
        if not enabled:
            return hidden_states
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError('memory expects hidden states with shape [batch, sequence, hidden_size]')
        work_dtype = torch.float32 if hidden_states.dtype in (torch.float16, torch.bfloat16) else hidden_states.dtype
        keys = self.K.to(work_dtype)
        values = self.V.detach().to(work_dtype) if stop_content_grad else self.V.to(work_dtype)
        output = self.W_O.detach().to(work_dtype) if stop_content_grad else self.W_O.to(work_dtype)
        # Qwen3.5 is normally bf16 on CUDA.  SDPA fuses the softmax and the
        # two memory matmuls, while the old path materialized a full
        # [batch, sequence_chunk, slots] float32 score tensor in Python.
        # Keep the RMS normalization in FP32, but use the model dtype for the
        # fused attention kernel.  CPU/FP32 callers retain the reference path.
        use_sdpa = (hidden_states.is_cuda and hidden_states.dtype in (torch.float16, torch.bfloat16)
                    and hasattr(F, 'scaled_dot_product_attention'))
        chunk_size = min(self.compute_chunk_size, hidden_states.shape[1])

        def project(chunk, mask):
            normalized = chunk.to(work_dtype)
            normalized = normalized * normalized.square().mean(dim=-1, keepdim=True).add(self.rms_eps).rsqrt()
            query = normalized.matmul(self.W_Q.to(work_dtype))
            if use_sdpa:
                kernel_dtype = hidden_states.dtype
                query = query.to(kernel_dtype).unsqueeze(1)
                key = keys.to(kernel_dtype).unsqueeze(0).unsqueeze(0)
                value = values.to(kernel_dtype).unsqueeze(0).unsqueeze(0)
                attended = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, scale=self.scale).squeeze(1)
                residual = self.dropout(attended.matmul(output.to(kernel_dtype))).to(hidden_states.dtype)
            else:
                weights = torch.softmax(query.matmul(keys.transpose(0, 1)) * self.scale, dim=-1)
                residual = self.dropout(weights.matmul(values).matmul(output)).to(hidden_states.dtype)
            if mask is not None:
                residual = residual * mask.to(dtype=residual.dtype).unsqueeze(-1)
            return chunk + residual

        chunks = []
        for start in range(0, hidden_states.shape[1], chunk_size):
            stop = min(start + chunk_size, hidden_states.shape[1])
            chunk = hidden_states[:, start:stop]
            mask = None
            if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[:2] == hidden_states.shape[:2]:
                mask = attention_mask[:, start:stop]
            if self.training and torch.is_grad_enabled() and hidden_states.shape[1] > chunk_size:
                chunks.append(checkpoint(project, chunk, mask, use_reentrant=False))
            else:
                chunks.append(project(chunk, mask))
        return torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
