import torch
import torch.nn.functional as F
import torch.nn as nn
import math

# Absolute positional encoding
class PositionEncoding(nn.Module):
    def __init__(self, d_model:int, max_len: int, base: float = 10000.0):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)
        inv_freq = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base)) / d_model)
        
        pe[:, 0::2] = torch.sin(torch.outer(position, inv_freq))
        pe[:, 1::2] = torch.cos(torch.outer(position, inv_freq))
        
        self.register_buffer('pe', pe)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.pe[t.long()]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        # scale and shift
        self.time_mlp = nn.Linear(time_emb_dim, out_channels * 2)
        
        # height + 2 * padding - kernel_size + 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else (
            nn.Identity()
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.bn1(x)
        out = self.act1(out)        
        out = self.conv1(x)
        
        # [batch_size, 2 * out_channels, 1, 1]
        time_emb = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = time_emb.chunk(2, dim=1)
        out = out * (1 + scale) + shift
        
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)
         
        out += identity # pre-activation x_l_add_1 = x_l + f(x_l)
        
        return out

class ConvNet(nn.Module):
    def __init__(
        self,
        n_steps,
        in_channels,
        out_channels,
        intermediate_channels = [10, 20, 40],
        pe_dim = 10,
    ):
        super().__init__()
              
        self.pe = PositionEncoding(pe_dim, n_steps)
        
        self.pe_projs = nn.ModuleList()
        
        self.residual_blocks = nn.ModuleList()
        
        prev_channel = in_channels
        for channel in intermediate_channels:
            self.residual_blocks.append(ResidualBlock(prev_channel, channel))
            prev_channel = channel
        
        self.o_proj = nn.Conv2d(prev_channel, out_channels, 3, 1, 1)
    
    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        t_emb = self.pe(t)
        
        for block in self.residual_blocks:
            x = block(x, t_emb)
        
        x = self.o_proj(x)
        
        return x