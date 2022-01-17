import math

import torch
import torch.nn as nn

from models.blocks import Swish


class TimeEmbedding(nn.Module):

    def __init__(self, n_channels: int):
        super(TimeEmbedding, self).__init__()
        self.n_channels = n_channels

        self.linear_1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.linear_2 = nn.Linear(self.n_channels, self.n_channels)
        self.act = Swish()

    def forward(self, t: torch.Tensor):
        # Paper(5page): using the Transformer sinusoidal position embedding.
        half_dim = self.n_channels // 8
        embedding = math.log(10_000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=t.device) * -embedding)
        embedding = t[:, None] * embedding[None, :] # todo print shape
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=1)

        embedding = self.act(self.linear_1(embedding))
        embedding = self.linear_2(embedding)

        return embedding
