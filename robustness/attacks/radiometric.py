"""
A4 (Global radiometric attack) and A5 (Random smooth radiometric control) --
spec sections 4.5-4.6.

A4:  I_adv = clip(alpha * I + beta),  |alpha-1| <= eps_alpha, |beta| <= eps_beta
     Deliberately low-capacity (2 scalars per image): "the bridge between
     generic corruption and spatial ARIA" (spec 4.5).

A5:  Mandatory random control for ARIA (A6). Per spec 4.6: "Generate a smooth
     multiplicative field with EXACTLY THE SAME AMPLITUDE CONSTRAINTS used by
     ARIA" -- i.e. the same coarse-grid + bilinear-upsample + tanh-bound
     parameterization as ARIA, with z sampled randomly instead of optimized.
     Implemented here (not in aria.py) because it is a *radiometric control*,
     but it shares aria._make_field with ARIAAttack to guarantee bounds/
     dimensionality match exactly (spec section 14 requirement).
"""
from typing import Optional

import torch

from ..core import AttackResult, BaseAttack, detection_loss
from ..severities import RADIOMETRIC_SEVERITIES, ARIA_SEVERITIES


class GlobalRadiometricAttack(BaseAttack):
    """A4: I_adv = clip(alpha*I + beta), optimized via sign-PGD on (alpha, beta)."""

    name = "global_radiometric"

    def __init__(self, eps_alpha, step_alpha, eps_beta, step_beta, steps=20, random_start=True):
        self.eps_alpha, self.step_alpha = eps_alpha, step_alpha
        self.eps_beta, self.step_beta = eps_beta, step_beta
        self.steps = steps
        self.random_start = random_start

    @classmethod
    def from_severity_name(cls, name: str, steps: int = 20, random_start: bool = True) -> "GlobalRadiometricAttack":
        s = RADIOMETRIC_SEVERITIES[name]
        return cls(s["eps_alpha"], s["step_alpha"], s["eps_beta"], s["step_beta"], steps=steps, random_start=random_start)

    def _generate(self, detector, batch, gen: Optional[torch.Generator]) -> AttackResult:
        x = batch["img"].detach()
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        if self.random_start:
            if gen is not None:
                da = (torch.rand((B, 1, 1, 1), generator=gen, dtype=dtype) * 2 - 1).to(device) * self.eps_alpha
                db = (torch.rand((B, 1, 1, 1), generator=gen, dtype=dtype) * 2 - 1).to(device) * self.eps_beta
            else:
                da = torch.empty((B, 1, 1, 1), device=device, dtype=dtype).uniform_(-self.eps_alpha, self.eps_alpha)
                db = torch.empty((B, 1, 1, 1), device=device, dtype=dtype).uniform_(-self.eps_beta, self.eps_beta)
        else:
            da = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)
            db = torch.zeros((B, 1, 1, 1), device=device, dtype=dtype)

        for _ in range(self.steps):
            da.requires_grad_(True)
            db.requires_grad_(True)
            alpha = 1.0 + da
            beta = db
            x_adv = torch.clamp(alpha * x + beta, 0.0, 1.0)
            loss = detection_loss(detector, x_adv, batch)
            grad_da, grad_db = torch.autograd.grad(loss, [da, db], retain_graph=False, create_graph=False, only_inputs=True)
            da = (da.detach() + self.step_alpha * grad_da.sign()).clamp(-self.eps_alpha, self.eps_alpha)
            db = (db.detach() + self.step_beta * grad_db.sign()).clamp(-self.eps_beta, self.eps_beta)

        alpha, beta = 1.0 + da, db
        x_adv = torch.clamp(alpha * x + beta, 0.0, 1.0).detach()
        return AttackResult(
            adv_images=x_adv,
            delta_or_params={"alpha": alpha.detach(), "beta": beta.detach()},
            attack_name=self.name,
            meta={"eps_alpha": self.eps_alpha, "eps_beta": self.eps_beta, "steps": self.steps,
                  "random_start": self.random_start, "adversarial": True},
        )


def make_aria_field(z: torch.Tensor, H: int, W: int, eps_field: float) -> torch.Tensor:
    """Shared ARIA field construction (used by both ARIAAttack and this
    module's random control) -- guarantees identical bounds/dimensionality
    per spec section 14. z: [B,1,h,w] coarse latent grid -> A: [B,1,H,W]."""
    U = torch.nn.functional.interpolate(z, size=(H, W), mode="bilinear", align_corners=False)
    A = 1.0 + eps_field * torch.tanh(U)
    return A


class RandomSmoothRadiometricAttack(BaseAttack):
    """A5: MANDATORY random control for ARIA (A6). Same coarse-grid + bilinear
    upsample + tanh-bound field as ARIA, z sampled randomly (not optimized)."""

    name = "random_radiometric"

    def __init__(self, eps_field: float, grid_size: int = 8):
        self.eps_field = eps_field
        self.grid_size = grid_size

    @classmethod
    def from_severity_name(cls, name: str, grid_size: int = 8) -> "RandomSmoothRadiometricAttack":
        return cls(ARIA_SEVERITIES[name].eps_field, grid_size=grid_size)

    def _generate(self, detector, batch, gen: Optional[torch.Generator]) -> AttackResult:
        x = batch["img"]
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        gh = gw = self.grid_size

        if gen is not None:
            z = (torch.rand((B, 1, gh, gw), generator=gen, dtype=dtype) * 2 - 1).to(device)
        else:
            z = torch.empty((B, 1, gh, gw), device=device, dtype=dtype).uniform_(-1, 1)

        A = make_aria_field(z, H, W, self.eps_field)
        x_adv = torch.clamp(A * x, 0.0, 1.0)

        return AttackResult(
            adv_images=x_adv,
            delta_or_params={"field_latent": z, "field_upsampled": A},
            attack_name=self.name,
            meta={"eps_field": self.eps_field, "grid_size": self.grid_size, "adversarial": False},
        )
