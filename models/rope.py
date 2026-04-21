import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class RoPE(nn.Module):
    def __init__(self, d_model: int, max_len: int, base: float = 10000.0):
        super().__init__()
        # self.dropout = nn.Dropout2d()
        
        # [0, 1, 2, ..., max_len - 1]
        position = torch.arange(0, max_len, dtype=torch.float)
        
        # 1 / (base ^ (2i / d_model)) = exp((2i / d_model) * -ln(base))
        inv_freq = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base)) / d_model)
        
        freqs = torch.outer(position, inv_freq) # [max_len, d_model // 2]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        return (x * cos) + (self._rotate_half(x) * sin)