from typing import Union, List, Tuple

import torch
import torch.nn as nn

from models.blocks import Swish
from models.embedding import TimeEmbedding
from models.blocks import ResidualBlock, AttentionBlock, DownBlock, UpBlock, MiddleBlock


class Downsample(nn.Module):

    def __init__(self, n_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Upsample(nn.Module):

    def __init__(self, n_channels):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, img_channels=3, n_channels=64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks=2):
        super(UNet, self).__init__()
        n_resolutions = len(ch_mults)

        # Mapping to Feature Space
        self.proj = nn.Conv2d(img_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        # Time Embedding Layer
        self.time_embedding = TimeEmbedding(n_channels * 4)

        # Unet Encoder
        down = []
        out_channels = in_channels = n_channels

        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]

            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1: # 마지막만 빼고 모두 Resolution DownSampling
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)
        print("down: ", self.down)
        # Unet Bridge
        self.middle = MiddleBlock(out_channels, n_channels * 4)

        # Unet Decoder
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels

            if i > 0: # 마지막만 빼고 모두 Resolution UpSampling
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)
        print("A", self.up)

        # Last Conv Layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, img_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        """
        inputs: [Batch, in_channels, h, w]
        t: [Batch]
        """
        t = self.time_embedding(t)

        # Get image projection
        outputs = self.proj(inputs)

        # `h` will store outputs at each resolution for skip connection
        encoder = [outputs]
        # First half of U-Net
        for down in self.down:
            outputs = down(outputs, t)
            encoder.append(outputs)
        print("A", len(encoder))
        # Middle (bottom)
        outputs = self.middle(outputs, t)

        # Second half of U-Net
        i = 0
        for up in self.up:
            if isinstance(up, Upsample):
                outputs = up(outputs, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                enc_out = encoder.pop()
                print(i, "e", enc_out.size(), "u", outputs.size())
                outputs = torch.cat([outputs, enc_out], dim=1)
                outputs = up(outputs, t)
            i += 1

        # Final normalization and convolution
        return self.final(self.act(self.norm(outputs)))
