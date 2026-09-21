"""
A1: Random fixed-pattern non-uniformity control (spec section 4.2).

    I'_{b,c,i,j} = (1 + g_{b,j}) * I_{b,c,i,j} + b_{b,j}

g, b sampled uniformly (NOT optimized) with the same bounds, dimensionality,
and broadcast structure as SNUA-GO -- this is the MANDATORY random control
spec section 14 requires for every structured SWIR attack, matched on:
parameter bounds, parameter dimensionality, image clipping, severity
setting, preprocessing, evaluation model. The only difference from SNUAGoAttack
is that these parameters are drawn from U(-eps, eps) once and never
optimized against the detector.
"""
from typing import Optional

import torch

from ..core import AttackResult, BaseAttack
from ..severities import SNUA_SEVERITIES, SNUASeverity


class RandomFPNAttack(BaseAttack):
    name = "random_fpn"

    def __init__(self, severity: SNUASeverity):
        self.severity = severity

    @classmethod
    def from_severity_name(cls, name: str) -> "RandomFPNAttack":
        return cls(SNUA_SEVERITIES[name])

    def _generate(self, detector, batch, gen: Optional[torch.Generator]) -> AttackResult:
        x = batch["img"]  # [B, C, H, W]
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        eps_g, eps_b = self.severity.eps_gain, self.severity.eps_offset

        if gen is not None:
            g = (torch.rand((B, 1, 1, W), generator=gen, dtype=dtype) * 2 - 1).to(device) * eps_g
            b = (torch.rand((B, 1, 1, W), generator=gen, dtype=dtype) * 2 - 1).to(device) * eps_b
        else:
            g = torch.empty((B, 1, 1, W), device=device, dtype=dtype).uniform_(-eps_g, eps_g)
            b = torch.empty((B, 1, 1, W), device=device, dtype=dtype).uniform_(-eps_b, eps_b)

        x_adv = torch.clamp((1 + g) * x + b, 0.0, 1.0)

        return AttackResult(
            adv_images=x_adv,
            delta_or_params={"gain": g, "offset": b},
            attack_name=self.name,
            meta={
                "eps_gain": eps_g, "eps_offset": eps_b,
                "shared_across_channels": True, "parameterization": "column", "adversarial": False,
            },
        )
