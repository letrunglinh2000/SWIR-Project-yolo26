"""
Common attack framework for the SWIR YOLO26 adversarial robustness track.

Every attack (generic PGD, Random FPN control, SNUA-GO, Smooth SNUA, Global
radiometric, Random smooth radiometric control, ARIA, Combined SNUA+ARIA)
implements the same `BaseAttack.generate()` contract, so the evaluation
runner, the unit-test suite, and the minimal-experiment harness are written
once against the interface rather than once per attack.

Research distinction this module exists to preserve (spec section 0):

    Generic attack:      I_adv = I + delta                (unconstrained per-pixel)
    SWIR-specific attack: I_adv = T_SWIR(I; p), p in C_SWIR (small physically
                          interpretable parameter set, e.g. per-column gain/
                          offset or a coarse radiometric field)

`BaseAttack` does not care which family a subclass belongs to; the
distinction is enforced by what each subclass optimizes (raw pixels vs. a
low-dimensional parameter vector) and is documented per-attack in
`robustness/attacks/`.
"""
import dataclasses
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG_DICT


def prepare_detector_for_attack(detector: nn.Module) -> nn.Module:
    """Convert a loaded checkpoint's `detector.args` from a raw dict into the
    `ultralytics.cfg` Namespace the loss function actually reads.

    Gotcha (confirmed empirically, see ROBUSTNESS_AUDIT_M0.json): a YOLO
    checkpoint's `model.args` is saved as a plain dict. Calling
    `detector.forward(batch_dict)` in train mode (the only way to reach the
    differentiable pre-NMS loss) crashes with
    `AttributeError: 'dict' object has no attribute 'box'` because the loss
    function reads `self.hyp.box` / `.cls` / `.dfl` as attributes. This was
    never hit by the project's earlier eval-only PGD scripts because those
    only ever called `forward(img_tensor)` in eval mode (inference, not
    loss). Idempotent: no-op if `detector.args` is already a Namespace.
    """
    if isinstance(detector.args, dict):
        saved_args = {k: v for k, v in detector.args.items() if k in DEFAULT_CFG_DICT}
        detector.args = get_cfg(overrides=saved_args)
    return detector


class DetectorLossContext:
    """Context manager that puts a YOLO detector into the exact mode needed
    to get a differentiable, pre-NMS detection loss with respect to its
    *input image*, without ever updating a model parameter, and restores the
    original mode on exit.

    Concretely: `.train()` (required -- `.eval()` mode returns decoded
    predictions, not the loss) with every BatchNorm layer forced back to
    `.eval()` (so attack optimization does not perturb BN running
    statistics), and every parameter's `requires_grad` forced to False for
    the duration (so `torch.autograd.grad` calls against the image can never
    accidentally also compute -- let alone apply -- a weight gradient).

    This mirrors the mode-juggling already present ad hoc in the project's
    original `train_attack.py` / `test_attack.py` PGD implementation, pulled
    into one reusable, unit-tested utility instead of being re-derived (and
    potentially re-broken) by every new attack.
    """

    def __init__(self, detector: nn.Module):
        self.detector = detector

    def __enter__(self) -> nn.Module:
        prepare_detector_for_attack(self.detector)
        self._orig_mode = self.detector.training
        self._bn_layers = [
            m for m in self.detector.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)
        ]
        self._bn_states = [m.training for m in self._bn_layers]
        self._param_requires_grad = [p.requires_grad for p in self.detector.parameters()]

        self.detector.train()
        for m in self._bn_layers:
            m.eval()
        for p in self.detector.parameters():
            p.requires_grad_(False)
        return self.detector

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        for p, req in zip(self.detector.parameters(), self._param_requires_grad):
            p.requires_grad_(req)
        for m, s in zip(self._bn_layers, self._bn_states):
            m.train(s)
        self.detector.train(self._orig_mode)
        return False


def detection_loss(detector: nn.Module, images: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
    """Differentiable, pre-NMS YOLO detection loss for a batch of images.

    `batch` must already be the ultralytics-preprocessed batch dict (labels,
    batch_idx, cls, bboxes, ...). `images` is substituted in as `batch['img']`
    via a *shallow copy* of `batch` -- the original dict (and therefore its
    label tensors) is never mutated, so callers can safely reuse `batch`
    across many attack iterations or many attacks on the same data.

    Returns the scalar total loss (summed over the 3 YOLO loss components --
    box, cls, dfl -- exactly as the model's own `.loss()` combines them; see
    `ultralytics.utils.loss` for the underlying weighted sum).
    """
    local_batch = dict(batch)
    local_batch["img"] = images
    out = detector(local_batch)
    loss = out[0] if isinstance(out, (tuple, list)) else out
    return loss.sum()


@dataclasses.dataclass
class AttackResult:
    """Standard return type for every `BaseAttack.generate()` call."""

    adv_images: torch.Tensor            # perturbed images, still in [0,1]
    delta_or_params: Dict[str, Any]     # attack-specific: raw pixel delta (PGD),
                                         # {gain, offset} (SNUA family), {grid} (ARIA), etc.
    attack_name: str
    severity: Optional[str] = None
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)  # steps, final loss, config, seed used


class BaseAttack:
    """Common interface every attack in this framework implements.

    Subclasses implement `_generate(detector, batch, gen)`, where `gen` is an
    optional seeded `torch.Generator` for reproducible stochastic attacks
    (random-start PGD, the Random FPN / Random radiometric *controls*).
    `generate()` wraps `_generate` with `DetectorLossContext` so individual
    attack implementations never have to re-derive the train/eval/BN/
    requires_grad mode-juggling themselves -- and, critically, so that
    forgetting it in a new attack cannot silently update model weights.
    """

    name: str = "base"

    def generate(
        self,
        detector: nn.Module,
        batch: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> AttackResult:
        if seed is not None:
            # Always a CPU generator: torch.Generator(device='cuda') support is
            # version-dependent, and sampling on CPU then .to(device) makes the
            # random draw itself reproducible independent of which GPU runs it.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
        else:
            gen = None
        with DetectorLossContext(detector):
            return self._generate(detector, batch, gen)

    def _generate(self, detector: nn.Module, batch: Dict[str, Any], gen: Optional[torch.Generator]) -> AttackResult:
        raise NotImplementedError
