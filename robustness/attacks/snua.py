"""
A2 (SNUA-GO) and A3 (Smooth SNUA) -- SWIR Non-Uniformity Attack, column
gain/offset version (spec sections 4.3-4.4).

    I_adv(i,j) = (1 + g_j) * I(i,j) + b_j

Optimized via projected sign-gradient ascent on (g, b):

    g_{t+1} = Pi_[-eps_g,eps_g][g_t + alpha_g * sign(grad_g L)]
    b_{t+1} = Pi_[-eps_b,eps_b][b_t + alpha_b * sign(grad_b L)]

`column` parameterization only. Spec section 4.3 designates column as "the
primary paper mode" and lists row as an "optional diagnostic mode", pixel
as "diagnostic only", and lowrank as an "optional later extension" -- none
are marked out of scope, but none are implemented in this first release
either (deferred as optional/diagnostic per that same wording).
Gain/offset are shared across channels by construction (broadcast
over the channel dim) -- required by spec section 4.2 for a single-band
SWIR image replicated to 3 channels, and verified by the channel-check unit
test (test_snua_unit.py::test_channel_sharing).

SmoothSNUAAttack (A3) adds the section-4.4 total-variation penalty on (g, b)
and maximizes L_det - lambda_smooth * L_smooth (NOT L_det + lambda * L_smooth
-- the attack is PENALIZED for roughness, never rewarded for it).
"""
from typing import Optional

import torch

from ..core import AttackResult, BaseAttack, detection_loss
from ..severities import SNUA_SEVERITIES, SNUASeverity


def _tv_smoothness_loss(g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Section 4.4: mean |g_{j+1}-g_j| + mean |b_{j+1}-b_j| along the column axis.
    g, b have shape [B,1,1,W]; the difference is taken along W (last dim)."""
    W = g.shape[-1]
    if W < 2:
        return torch.zeros((), device=g.device, dtype=g.dtype)
    tv_g = (g[..., 1:] - g[..., :-1]).abs().mean()
    tv_b = (b[..., 1:] - b[..., :-1]).abs().mean()
    return tv_g + tv_b


def _init_params(x: torch.Tensor, severity: SNUASeverity, gen: Optional[torch.Generator], random_start: bool):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    if random_start:
        if gen is not None:
            g = (torch.rand((B, 1, 1, W), generator=gen, dtype=dtype) * 2 - 1).to(device) * severity.eps_gain
            b = (torch.rand((B, 1, 1, W), generator=gen, dtype=dtype) * 2 - 1).to(device) * severity.eps_offset
        else:
            g = torch.empty((B, 1, 1, W), device=device, dtype=dtype).uniform_(-severity.eps_gain, severity.eps_gain)
            b = torch.empty((B, 1, 1, W), device=device, dtype=dtype).uniform_(-severity.eps_offset, severity.eps_offset)
    else:
        g = torch.zeros((B, 1, 1, W), device=device, dtype=dtype)
        b = torch.zeros((B, 1, 1, W), device=device, dtype=dtype)
    return g, b


def _apply(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp((1 + g) * x + b, 0.0, 1.0)


class SNUAGoAttack(BaseAttack):
    """A2: SNUA-GO -- main SWIR sensor adversarial attack."""

    name = "snua_go"

    def __init__(self, severity: SNUASeverity, random_start: bool = True):
        self.severity = severity
        self.random_start = random_start

    @classmethod
    def from_severity_name(cls, name: str, random_start: bool = True) -> "SNUAGoAttack":
        return cls(SNUA_SEVERITIES[name], random_start=random_start)

    def _generate(self, detector, batch, gen: Optional[torch.Generator]) -> AttackResult:
        x = batch["img"].detach()
        s = self.severity

        if s.steps == 0:
            zeros = torch.zeros_like(x[:, :1, :1, :])
            return AttackResult(
                adv_images=x.clone(), delta_or_params={"gain": zeros, "offset": zeros},
                attack_name=self.name, meta={"steps": 0},
            )

        g, b = _init_params(x, s, gen, self.random_start)

        for _ in range(s.steps):
            g.requires_grad_(True)
            b.requires_grad_(True)
            x_adv = _apply(x, g, b)
            loss = detection_loss(detector, x_adv, batch)
            grad_g, grad_b = torch.autograd.grad(loss, [g, b], retain_graph=False, create_graph=False, only_inputs=True)
            g = (g.detach() + s.step_gain * grad_g.sign()).clamp(-s.eps_gain, s.eps_gain)
            b = (b.detach() + s.step_offset * grad_b.sign()).clamp(-s.eps_offset, s.eps_offset)

        x_adv = _apply(x, g, b).detach()
        return AttackResult(
            adv_images=x_adv,
            delta_or_params={"gain": g.detach(), "offset": b.detach()},
            attack_name=self.name,
            meta={
                "eps_gain": s.eps_gain, "eps_offset": s.eps_offset, "steps": s.steps,
                "random_start": self.random_start, "adversarial": True,
            },
        )


class SmoothSNUAAttack(BaseAttack):
    """A3: Smooth SNUA -- SNUA-GO with a TV-smoothness penalty on (g, b)."""

    name = "smooth_snua"

    def __init__(self, severity: SNUASeverity, lambda_smooth: float = 1.0, random_start: bool = True):
        self.severity = severity
        self.lambda_smooth = lambda_smooth
        self.random_start = random_start

    @classmethod
    def from_severity_name(cls, name: str, lambda_smooth: float = 1.0, random_start: bool = True) -> "SmoothSNUAAttack":
        return cls(SNUA_SEVERITIES[name], lambda_smooth=lambda_smooth, random_start=random_start)

    def _generate(self, detector, batch, gen: Optional[torch.Generator]) -> AttackResult:
        x = batch["img"].detach()
        s = self.severity
        g, b = _init_params(x, s, gen, self.random_start)

        last_det_loss = None
        last_smooth_loss = None
        for _ in range(s.steps):
            g.requires_grad_(True)
            b.requires_grad_(True)
            x_adv = _apply(x, g, b)
            det_loss = detection_loss(detector, x_adv, batch)
            smooth_loss = _tv_smoothness_loss(g, b)
            objective = det_loss - self.lambda_smooth * smooth_loss  # maximize det_loss, penalize roughness
            grad_g, grad_b = torch.autograd.grad(objective, [g, b], retain_graph=False, create_graph=False, only_inputs=True)
            g = (g.detach() + s.step_gain * grad_g.sign()).clamp(-s.eps_gain, s.eps_gain)
            b = (b.detach() + s.step_offset * grad_b.sign()).clamp(-s.eps_offset, s.eps_offset)
            last_det_loss = det_loss.item()
            last_smooth_loss = smooth_loss.item()

        x_adv = _apply(x, g, b).detach()
        return AttackResult(
            adv_images=x_adv,
            delta_or_params={"gain": g.detach(), "offset": b.detach()},
            attack_name=self.name,
            meta={
                "eps_gain": s.eps_gain, "eps_offset": s.eps_offset, "steps": s.steps,
                "lambda_smooth": self.lambda_smooth, "random_start": self.random_start,
                "final_det_loss": last_det_loss, "final_smooth_loss": last_smooth_loss, "adversarial": True,
            },
        )
