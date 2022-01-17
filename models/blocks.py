from typing import Optional

import torch
import torch.nn as nn


class Swish(nn.Module):

    def forward(self, inputs):
        return inputs * torch.sigmoid(inputs)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        super(ResidualBlock, self).__init__()

        self.norm_1 = nn.GroupNorm(n_groups, in_channels)
        self.act_1 = Swish()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.norm_2 = nn.GroupNorm(n_groups, out_channels)
        self.act_2 = Swish()
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.time_embedding = nn.Linear(time_channels, out_channels)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        """
        inputs: [batch, in_channels, h, w]
        t: [batch, time_channels]
        """
        out = self.conv_1(self.act_1(self.norm_1(inputs)))
        out += self.time_embedding(t)[:, :, None, None]
        out = self.conv_2(self.act_2(self.norm_2(out)))

        return out + self.shortcut(inputs)


class AttentionBlock(nn.Module):

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super(AttentionBlock, self).__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res


class DownBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super(DownBlock, self).__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        outputs = self.res(inputs, t)
        outputs = self.attn(outputs)
        return outputs


class UpBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super(UpBlock, self).__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        outputs = self.res(inputs, t)
        outputs = self.attn(outputs)
        return outputs


class MiddleBlock(nn.Module):

    def __init__(self, n_channels: int, time_channels: int):
        super(MiddleBlock, self).__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        outputs = self.res1(inputs, t)
        outputs = self.attn(outputs)
        outputs = self.res2(outputs, t)
        return outputs