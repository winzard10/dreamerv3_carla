import math
from typing import Callable, Optional

import torch


class LaProp(torch.optim.Optimizer):
    """
    Lightweight LaProp optimizer implementation compatible with Python 3.7.

    Update rule (bias-corrected):
      v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
      m_t = beta1 * m_{t-1} + (1-beta1) * (g_t / (sqrt(v_t) + eps))
      p_t = p_{t-1} - lr * sqrt(1-beta2^t)/(1-beta1^t) * m_t

    Supports decoupled weight decay (AdamW-style).
    """

    def __init__(
        self,
        params,
        lr=4e-4,
        betas=(0.9, 0.999),
        eps=1e-15,
        weight_decay=0.0,
        weight_decouple=True,
        fixed_decay=False,
        maximize=False,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if len(betas) != 2:
            raise ValueError("betas must be a tuple of length 2")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            fixed_decay=fixed_decay,
            maximize=maximize,
        )
        super(LaProp, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            weight_decouple = group["weight_decouple"]
            fixed_decay = group["fixed_decay"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")
                if maximize:
                    grad = -grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Weight decay
                if weight_decay != 0.0:
                    if weight_decouple:
                        if fixed_decay:
                            p.mul_(1.0 - weight_decay)
                        else:
                            p.mul_(1.0 - lr * weight_decay)
                    else:
                        grad = grad.add(p, alpha=weight_decay)

                # Second moment
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)

                # LaProp momentum on preconditioned gradient
                precond_grad = grad / denom
                exp_avg.mul_(beta1).add_(precond_grad, alpha=1.0 - beta1)

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                p.add_(exp_avg, alpha=-step_size)

        return loss
