"""
https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/8af24da2dd39a9a87482a4d18c2dc829bbd3fd47/labml_nn/diffusion/ddpm/__init__.py#L282
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gather import gather


class DenoiseDiffusion(nn.Module):

    def __init__(self, eps_model: nn.Module, n_steps: int, args):
        super().__init__()
        self.eps_model = eps_model # U_Net
        self.n_steps = n_steps

        # Variance Scheduler
        # torch.linspace: (start, start+ (end-start)/(steps-1), ..., end)
        self.beta = torch.linspace(start=0.0001, end=0.02, steps=self.n_steps).cuda(args.gpu)
        self.alpha = 1. - self.beta # paper(2page, Eq.4)
        # torch.cumprod: Returns the cumulative product of elemetns of input in the dimension dim.
        # y_i = x_1 * x_2 * x_3 ... * x_i
        self.alpha_bar = torch.cumprod(self.alpha, dim=0) # alpha_bar = product{s=1}(alpha_s
        self.sigma2 = self.beta # sigma**2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process(Diffusion Process).
        """
        # todo x0, t가 무엇인지 확인
        mean = gather(consts=self.alpha_bar, t=t) ** 0.5 * x0
        var = 1 - gather(consts=self.alpha_bar, t=t)

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor]=None):
        """
        Sample from q(x_t|x_0)
        """
        # todo Input이 무엇인지 확인
        if eps is None:
            eps = torch.rand_like(x0)

        mean, var = self.q_xt_x0(x0=x0, t=t) # q distribution 얻기

        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Sample from p_theta(x_t-1| x_t)
        """
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)

        eps_coefficient = (1 - alpha) / (1 - alpha_bar) ** .5
        # paper(4page): u_theta must predict (1/alpha_t) * (x_t - beta_t/(sqrt(1-alpha_bar_t)) * eps)

        mean = 1 / (alpha ** 0.5) * (xt - eps_coefficient * eps_theta)
        # Paper(4page, Eq.11)
        var = gather(self.sigma2, t)

        eps = torch.randn(xt.shape, device=xt.device)

        return mean + (var ** .5) * eps

    def forward(self, inputs: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch = inputs.size(0)
        # 난수 생성
        t = torch.randint(0, self.n_steps, (batch, ), device=inputs.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(inputs)

        # Sample x_t for q(x_t|x_0)
        xt = self.q_sample(inputs, t, eps=noise)
        # eps_theta 구하기 (paper Eq.14)
        eps_theta = self.eps_model(xt, t)

        return noise, eps_theta



