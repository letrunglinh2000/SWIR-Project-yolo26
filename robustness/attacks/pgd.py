"""
Generic L-infinity PGD attack (Madry et al., 2018): I_adv = I + delta.

Ported into the common `BaseAttack` interface, with the SAME semantics as
this project's original ad hoc PGD implementation
(train_attack.py / test_attack.py / pgd_robustness_grid*.py) -- same random
start, same sign-gradient step, same per-step L-infinity projection and
[0,1] pixel clamp. This class is the M1 regression anchor: it must reproduce
the original script's clean/PGD-20 numbers within tolerance
(robustness/tests/test_pgd_regression.py) before any new SWIR-specific
attack built on this framework is trusted.
"""
from typing import Optional

import torch

from ..core import AttackResult, BaseAttack, detection_loss


class PGDAttack(BaseAttack):
    name = "pgd"

    def __init__(self, eps: float = 8 / 255, alpha: float = 2 / 255, steps: int = 20, random_start: bool = True):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def _generate(self, detector, batch, gen: Optional[torch.Generator]) -> AttackResult:
        x = batch["img"].detach()

        if self.steps == 0:
            # Identity case: no perturbation at all. Used by the framework's
            # sanity-check suite (zero-budget attack must return the image
            # bit-for-bit unchanged) and by any "clean" evaluation that wants
            # to go through the same code path as an attacked one.
            return AttackResult(
                adv_images=x.clone(),
                delta_or_params={"delta": torch.zeros_like(x)},
                attack_name=self.name,
                meta={"eps": self.eps, "alpha": self.alpha, "steps": 0, "random_start": self.random_start},
            )

        if self.random_start:
            if gen is not None:
                noise = torch.rand(x.shape, generator=gen, dtype=x.dtype) * 2 - 1
                delta = (noise.to(x.device) * self.eps)
            else:
                delta = torch.empty_like(x).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x + delta, 0.0, 1.0)
        else:
            x_adv = x.clone()
        x_adv = x_adv.detach()

        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            loss = detection_loss(detector, x_adv, batch)
            grad = torch.autograd.grad(
                outputs=loss, inputs=x_adv, retain_graph=False, create_graph=False, only_inputs=True
            )[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + delta, min=0.0, max=1.0).detach()

        final_delta = (x_adv - x).detach()
        return AttackResult(
            adv_images=x_adv,
            delta_or_params={"delta": final_delta},
            attack_name=self.name,
            meta={"eps": self.eps, "alpha": self.alpha, "steps": self.steps, "random_start": self.random_start},
        )
