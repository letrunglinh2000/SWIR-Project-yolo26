"""
A7: Combined SNUA + ARIA (spec section 4.8).

    I_adv = A ⊙ [(1+G) ⊙ I + B]

Primary variables: column gain G, column offset B (SNUA), coarse
radiometric field A (ARIA, via the shared `radiometric.make_aria_field`).
Constraints on G/B/A are each applied independently (same per-attack bounds
as SNUAGoAttack / ARIAAttack).

Joint mode implemented first, per spec ("Implement joint mode first. Add
alternating mode only if joint optimization is unstable or one component
dominates.") -- all three parameter groups (G, B, z) are updated from the
SAME detector loss every iteration via sign-PGD.

Note: the spec's section 4.8 text defines only the formula, constraints,
and the joint/alternating optimization modes -- it makes no claim about
combined-vs-component attack strength. The monotonicity check in
test_combined_unit.py (combined loss >= max(component losses)) is this
implementation's OWN sanity check, not a documented spec requirement.
"""
from typing import Optional

import torch

from ..core import AttackResult, BaseAttack, detection_loss
from ..severities import ARIA_SEVERITIES, SNUA_SEVERITIES, ARIASeverity, SNUASeverity
from .radiometric import make_aria_field


class CombinedSNUAARIAAttack(BaseAttack):
    """Joint-mode combined SNUA+ARIA attack."""

    name = "combined_snua_aria"

    def __init__(self, snua_severity: SNUASeverity, aria_severity: ARIASeverity,
                 grid_size: int = 8, random_start: bool = True):
        self.snua_severity = snua_severity
        self.aria_severity = aria_severity
        self.grid_size = grid_size
        self.random_start = random_start
        # joint mode runs a single shared step count -- the two severity
        # tables carry independent `steps` fields, so use the max to avoid
        # silently truncating whichever component would otherwise get fewer
        # optimization iterations.
        self.steps = max(snua_severity.steps, aria_severity.steps)

    @classmethod
    def from_severity_names(cls, snua_name: str, aria_name: str, grid_size: int = 8,
                             random_start: bool = True) -> "CombinedSNUAARIAAttack":
        return cls(SNUA_SEVERITIES[snua_name], ARIA_SEVERITIES[aria_name], grid_size=grid_size, random_start=random_start)

    def _generate(self, detector, batch, gen: Optional[torch.Generator]) -> AttackResult:
        x = batch["img"].detach()
        B_, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        s_snua, s_aria = self.snua_severity, self.aria_severity
        gh = gw = self.grid_size

        if self.random_start:
            if gen is not None:
                g = (torch.rand((B_, 1, 1, W), generator=gen, dtype=dtype) * 2 - 1).to(device) * s_snua.eps_gain
                b = (torch.rand((B_, 1, 1, W), generator=gen, dtype=dtype) * 2 - 1).to(device) * s_snua.eps_offset
                z = (torch.rand((B_, 1, gh, gw), generator=gen, dtype=dtype) * 2 - 1).to(device)
            else:
                g = torch.empty((B_, 1, 1, W), device=device, dtype=dtype).uniform_(-s_snua.eps_gain, s_snua.eps_gain)
                b = torch.empty((B_, 1, 1, W), device=device, dtype=dtype).uniform_(-s_snua.eps_offset, s_snua.eps_offset)
                z = torch.empty((B_, 1, gh, gw), device=device, dtype=dtype).uniform_(-1, 1)
        else:
            g = torch.zeros((B_, 1, 1, W), device=device, dtype=dtype)
            b = torch.zeros((B_, 1, 1, W), device=device, dtype=dtype)
            z = torch.zeros((B_, 1, gh, gw), device=device, dtype=dtype)

        def _apply(g_, b_, z_):
            snua_img = torch.clamp((1 + g_) * x + b_, 0.0, 1.0)
            A = make_aria_field(z_, H, W, s_aria.eps_field)
            return torch.clamp(A * snua_img, 0.0, 1.0)

        for _ in range(self.steps):
            g.requires_grad_(True)
            b.requires_grad_(True)
            z.requires_grad_(True)
            x_adv = _apply(g, b, z)
            loss = detection_loss(detector, x_adv, batch)
            grad_g, grad_b, grad_z = torch.autograd.grad(
                loss, [g, b, z], retain_graph=False, create_graph=False, only_inputs=True
            )
            g = (g.detach() + s_snua.step_gain * grad_g.sign()).clamp(-s_snua.eps_gain, s_snua.eps_gain)
            b = (b.detach() + s_snua.step_offset * grad_b.sign()).clamp(-s_snua.eps_offset, s_snua.eps_offset)
            z = (z.detach() + s_aria.step_field * grad_z.sign()).clamp(-4.0, 4.0)

        x_adv = _apply(g, b, z).detach()
        A_final = make_aria_field(z, H, W, s_aria.eps_field).detach()

        return AttackResult(
            adv_images=x_adv,
            delta_or_params={"gain": g.detach(), "offset": b.detach(), "field_latent": z.detach(), "field_upsampled": A_final},
            attack_name=self.name,
            meta={
                "eps_gain": s_snua.eps_gain, "eps_offset": s_snua.eps_offset, "eps_field": s_aria.eps_field,
                "steps": self.steps, "mode": "joint", "random_start": self.random_start, "adversarial": True,
            },
        )
