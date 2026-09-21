"""
A6: ARIA -- Adversarial Radiometric Illumination Attack (spec section 4.7).

    z in R^{B,1,h,w}  (coarse latent grid, h,w << H,W: 4x4/8x8/16x16)
    U = BilinearUpsample(z, H, W)
    A = 1 + eps_A * tanh(U)
    I_adv = clip(A ⊙ I)

Optimize z via sign-PGD to maximize the YOLO detection loss. The low-
resolution parameterization is deliberate (spec: "naturally suppresses
pixel-scale adversarial noise") -- gradients flow through the differentiable
bilinear upsample (`radiometric.make_aria_field`, shared with the A5 random
control to guarantee identical bounds/dimensionality per spec section 14).
"""
from typing import Optional

import torch

from ..core import AttackResult, BaseAttack, detection_loss
from ..severities import ARIA_SEVERITIES, ARIASeverity
from .radiometric import make_aria_field


class ARIAAttack(BaseAttack):
    name = "aria"

    def __init__(self, severity: ARIASeverity, grid_size: int = 8, random_start: bool = True):
        self.severity = severity
        self.grid_size = grid_size
        self.random_start = random_start

    @classmethod
    def from_severity_name(cls, name: str, grid_size: int = 8, random_start: bool = True) -> "ARIAAttack":
        return cls(ARIA_SEVERITIES[name], grid_size=grid_size, random_start=random_start)

    def _generate(self, detector, batch, gen: Optional[torch.Generator]) -> AttackResult:
        x = batch["img"].detach()
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        s = self.severity
        gh = gw = self.grid_size

        if s.steps == 0:
            zeros = torch.zeros((B, 1, gh, gw), device=device, dtype=dtype)
            return AttackResult(adv_images=x.clone(), delta_or_params={"field_latent": zeros},
                                 attack_name=self.name, meta={"steps": 0})

        if self.random_start:
            if gen is not None:
                z = (torch.rand((B, 1, gh, gw), generator=gen, dtype=dtype) * 2 - 1).to(device)
            else:
                z = torch.empty((B, 1, gh, gw), device=device, dtype=dtype).uniform_(-1, 1)
        else:
            z = torch.zeros((B, 1, gh, gw), device=device, dtype=dtype)

        for _ in range(s.steps):
            z.requires_grad_(True)
            A = make_aria_field(z, H, W, s.eps_field)
            x_adv = torch.clamp(A * x, 0.0, 1.0)
            loss = detection_loss(detector, x_adv, batch)
            grad_z = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False, only_inputs=True)[0]
            # z itself is NOT box-constrained (it's a pre-tanh latent) -- the
            # PHYSICAL constraint |A-1| <= eps_field is enforced by tanh
            # saturating to +-1 regardless of z's magnitude, but we still cap
            # z's raw step so tanh doesn't saturate into a degenerate
            # binary field after only a few steps (keeps the field genuinely
            # smooth/low-frequency rather than a step function).
            z = (z.detach() + s.step_field * grad_z.sign()).clamp(-4.0, 4.0)

        A = make_aria_field(z, H, W, s.eps_field)
        x_adv = torch.clamp(A * x, 0.0, 1.0).detach()

        return AttackResult(
            adv_images=x_adv,
            delta_or_params={"field_latent": z.detach(), "field_upsampled": A.detach()},
            attack_name=self.name,
            meta={"eps_field": s.eps_field, "grid_size": self.grid_size, "steps": s.steps,
                  "random_start": self.random_start, "adversarial": True},
        )
