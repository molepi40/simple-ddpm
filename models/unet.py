import torch
import torch.nn.functional as F
import torch.nn as nn

from .convnet import PositionEncoding


class UnetBlock(nn.Module):
    def __init__(self, shape, in_channels, out_channels, residual=False):
        super().__init__()
        
        self.ln = nn.LayerNorm(shape)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
        self.act = nn.ReLU()
        self.residual = residual
        
        if self.residual:
            self.residual_conv = (
                nn.Conv2d(in_channels, out_channels, 1)
            ) if in_channels != out_channels else (
                nn.Identity()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ln(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        
        if self.residual:
            out += self.residual_conv(x)
            
        out = self.act(out)
        
        return out


class UNet(nn.Module):
    def __init__(
        self,
        n_steps,
        in_channels,
        out_channels,
        intermediate_channels,
        height,
        width,
        n_classes=10,
        pe_dim=10,
        residual=False
    ):
        super().__init__()

        self.n_classes = n_classes
        self.null_label = n_classes
        
        num_blocks = len(intermediate_channels)
        Hs = [height] * num_blocks
        Ws = [width] * num_blocks
        cH, cW = height, width
        for i in range(1, num_blocks):
            cH, cW = (cH + 1) // 2, (cW + 1) // 2
            Hs[i] = cH
            Ws[i] = cW

        self.pe = PositionEncoding(pe_dim, n_steps)
        self.class_embedding = nn.Embedding(n_classes + 1, pe_dim)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.encoder_pe_projs = nn.ModuleList()
        self.decoder_pe_projs = nn.ModuleList()
        self.down_projs = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        
        prev_channel = in_channels
        for i, (channel, cH, cW) in enumerate(zip(intermediate_channels, Hs, Ws)):
            self.encoder_pe_projs.append(
                nn.Sequential(
                    nn.Linear(pe_dim, prev_channel),
                    nn.ReLU(),
                    nn.Linear(prev_channel, prev_channel)
                )
            )
            
            self.encoders.append(
                nn.Sequential(
                    UnetBlock(
                        (prev_channel, cH, cW),
                        prev_channel,
                        channel,
                        residual=residual
                    ),
                    UnetBlock(
                        (channel, cH, cW),
                        channel,
                        channel,
                        residual=residual
                    )
                )
            )
            
            if i < num_blocks - 1:
                self.down_projs.append(
                    nn.Conv2d(channel, channel, 2, 2)
                )
            
            prev_channel = channel
        
        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        channel = intermediate_channels[-1]
        self.mid = nn.Sequential(
            UnetBlock(
                (prev_channel, Hs[-1], Ws[-1]),
                prev_channel,
                channel,
                residual=residual
            ),
            UnetBlock(
                (channel, Hs[-1], Ws[-1]),
                channel,
                channel,
                residual=residual
            )
        )
        
        prev_channel = channel
        for channel, cH, cW in zip(intermediate_channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            self.decoder_pe_projs.append(
                nn.Linear(pe_dim, channel)
            )
            self.up_projs.append(
                nn.ConvTranspose2d(prev_channel, channel, 2, 2)
            )
            self.decoders.append(
                nn.Sequential(
                    UnetBlock(
                        (channel * 2, cH, cW),
                        channel * 2,
                        channel,
                        residual=residual
                    ),
                    UnetBlock(
                        (channel, cH, cW),
                        channel,
                        channel,
                        residual=residual
                    )
                )
            )
            
            prev_channel = channel
        
        self.o_proj = nn.Conv2d(prev_channel, out_channels, 3, 1, 1)
    
    def forward(self, x, t, y=None):
        n = t.shape[0]

        t_emb = self.pe(t)
        if y is None:
            y = torch.full((n,), self.null_label, device=t.device, dtype=torch.long)
        else:
            y = y.to(t.device, dtype=torch.long)

        c_emb = t_emb + self.class_embedding(y)

        encoder_outs = []
        for i, (pe_proj, encoder) in enumerate(zip(self.encoder_pe_projs, self.encoders)):
            pe = pe_proj(c_emb).reshape(n, -1, 1, 1)
            x = encoder(x + pe)
            encoder_outs.append(x)
            if i < len(self.down_projs):
                x = self.down_projs[i](x)
        
        pe = self.pe_mid(c_emb).reshape(n, -1, 1, 1)
        x = self.mid(x + pe)
        
        skip_outs = encoder_outs[:-1][::-1]
        for pe_proj, decoder, up_proj, encoder_out in zip(self.decoder_pe_projs,
                                                          self.decoders,
                                                          self.up_projs,
                                                          skip_outs):
            pe = pe_proj(c_emb).reshape(n, -1, 1, 1)
            x = up_proj(x) + pe
            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))
            x = decoder(torch.cat((x, encoder_out), dim=1))
        
        return self.o_proj(x)
                