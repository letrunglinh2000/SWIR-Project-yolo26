"""
Severity-level parameter tables for the SWIR-specific attacks (SNUA, ARIA).

Per spec section 12.1: at the software-development stage, use several small
normalized budgets and three severity levels (weak/medium/strong) WITHOUT
labeling them physically realistic yet -- that requires dark-frame/flat-field
sensor calibration (section 12.2), not yet performed. These are placeholders
for debugging and the minimal-experiment comparisons, not final scientific
values -- said explicitly in the spec's own example config (section 11).

The `medium` values below are taken directly from the spec's example YAML
configs (section 11): SNUA gain eps=0.05/step=0.005, offset eps=0.02/
step=0.002; ARIA field eps=0.10/step=0.01. `weak`/`strong` scale by 0.5x/2x.
In every case the observed ratio step_size/eps = 0.1 in the spec's own
example is preserved.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SNUASeverity:
    eps_gain: float
    step_gain: float
    eps_offset: float
    step_offset: float
    steps: int = 20


@dataclass(frozen=True)
class ARIASeverity:
    eps_field: float
    step_field: float
    steps: int = 20


SNUA_SEVERITIES = {
    "weak":   SNUASeverity(eps_gain=0.025, step_gain=0.0025, eps_offset=0.010, step_offset=0.0010),
    "medium": SNUASeverity(eps_gain=0.050, step_gain=0.0050, eps_offset=0.020, step_offset=0.0020),
    "strong": SNUASeverity(eps_gain=0.100, step_gain=0.0100, eps_offset=0.040, step_offset=0.0040),
}

ARIA_SEVERITIES = {
    "weak":   ARIASeverity(eps_field=0.05, step_field=0.005),
    "medium": ARIASeverity(eps_field=0.10, step_field=0.010),
    "strong": ARIASeverity(eps_field=0.20, step_field=0.020),
}

# Global radiometric (A4) uses the same amplitude family as ARIA medium by
# convention (both are "how far illumination is allowed to swing"); kept as
# its own table since alpha/beta are unconstrained-dimension (2 scalars)
# rather than a spatial field.
RADIOMETRIC_SEVERITIES = {
    "weak":   {"eps_alpha": 0.05, "eps_beta": 0.05, "step_alpha": 0.005, "step_beta": 0.005},
    "medium": {"eps_alpha": 0.10, "eps_beta": 0.10, "step_alpha": 0.010, "step_beta": 0.010},
    "strong": {"eps_alpha": 0.20, "eps_beta": 0.20, "step_alpha": 0.020, "step_beta": 0.020},
}
