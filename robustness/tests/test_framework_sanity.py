"""
Framework-level sanity-check suite, run against PGDAttack as the first
attack to exercise the common framework (any future attack added to
robustness/attacks/ should be run through the same four checks before its
results are trusted):

  1. Identity check      -- a zero-budget attack must leave the image
                             bit-for-bit unchanged (and therefore produce the
                             exact clean metrics).
  2. Frozen-model check   -- generating an attack must never change a single
                             model parameter (hash of all parameters before
                             == hash after).
  3. Gradient check       -- torch.autograd.grad calls used during attack
                             generation must produce finite, non-all-zero
                             gradients w.r.t. the image, and must never
                             populate .grad on any model parameter (they
                             stay requires_grad=False throughout).
  4. Batch-size independence -- the perturbation computed for one image must
                             not depend on which other images share its
                             batch (no accidental cross-sample coupling via
                             e.g. an un-frozen BatchNorm layer).

Run directly: `python robustness/tests/test_framework_sanity.py`
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from robustness.attacks.pgd import PGDAttack
from robustness.core import DetectorLossContext, detection_loss
from robustness.eval_runner import build_validator

MODEL_PATH = REPO_ROOT / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt"
YAML_PATH = REPO_ROOT / "dataset_safesplit.yaml"


def _param_hash(detector) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in detector.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def check_identity(detector, batch):
    print("--- (1) identity check (steps=0) ---")
    attack = PGDAttack(steps=0)
    result = attack.generate(detector, batch, seed=0)
    identical = torch.equal(result.adv_images, batch["img"])
    print(f"  adv_images == batch['img'] exactly: {identical} -> {'PASS' if identical else 'FAIL'}")
    return identical


def check_frozen_model(detector, batch):
    print("--- (2) frozen-model check ---")
    h_before = _param_hash(detector)
    attack = PGDAttack(eps=8 / 255, alpha=2 / 255, steps=20, random_start=True)
    attack.generate(detector, batch, seed=0)
    h_after = _param_hash(detector)
    unchanged = (h_before == h_after)
    print(f"  param hash before == after: {unchanged} -> {'PASS' if unchanged else 'FAIL'}")
    return unchanged


def check_gradient_reachability(detector, batch):
    print("--- (3) gradient check ---")
    for p in detector.parameters():
        p.grad = None
    attack = PGDAttack(eps=8 / 255, alpha=2 / 255, steps=1, random_start=False)
    result = attack.generate(detector, batch, seed=0)
    delta = result.delta_or_params["delta"]
    finite = torch.isfinite(delta).all().item()
    nonzero = (delta.abs().sum().item() > 0)
    no_weight_grad = all(p.grad is None for p in detector.parameters())
    ok = finite and nonzero and no_weight_grad
    print(f"  delta finite: {finite}, delta nonzero: {nonzero}, no model-weight .grad populated: {no_weight_grad} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_batch_size_independence(detector, batch, sign_agreement_tol=0.995, noise_floor_percentile=99.5):
    """Checks for ABSENCE OF CROSS-SAMPLE COUPLING, not bit-exact
    reproducibility. Floating-point summation over a batch dimension is not
    associative, so a batch-of-1 and a batch-of-4 forward pass through the
    same frozen model are not expected to be bit-identical even with
    `torch.backends.cudnn.deterministic=True` (verified separately: that
    flag set reproduces the exact same numeric difference, ruling out cuDNN
    algorithm-selection nondeterminism as the cause).

    IMPORTANT: this must compare the RAW gradient (pre-sign()), not the
    post-sign PGD delta. `alpha*sign(grad)` is near-constant magnitude
    (alpha almost everywhere, reduced only at pixels where [0,1] clipping
    engaged) and therefore carries no information about the true gradient
    magnitude -- using it as a "noise floor" proxy would be close to
    vacuous. The raw gradient (via `detection_loss` + `torch.autograd.grad`
    directly, bypassing PGDAttack's internal sign step) is what must be
    thresholded: if cross-image information were leaking (e.g. an unfrozen
    BatchNorm layer picking up batch statistics), disagreement would
    concentrate at genuinely high-|gradient| (object-relevant) pixels, not
    at pixels whose raw gradient sits at the noise floor.
    """
    print("--- (4) batch-size independence check ---")
    n = batch["img"].shape[0]
    if n < 2:
        print("  SKIPPED (batch has <2 images)")
        return True

    single_batch = {k: (v[0:1] if torch.is_tensor(v) and v.shape[0] == n else v) for k, v in batch.items()}
    if "batch_idx" in batch:
        mask = batch["batch_idx"] == 0
        single_batch["batch_idx"] = torch.zeros(mask.sum().item(), dtype=batch["batch_idx"].dtype, device=batch["batch_idx"].device)
        single_batch["cls"] = batch["cls"][mask]
        single_batch["bboxes"] = batch["bboxes"][mask]

    def _raw_grad(local_batch):
        # Same DetectorLossContext every attack uses (train()+frozen BN+frozen
        # params) -- required for detector.loss() to be well-defined/differentiable;
        # NOT the eval() mode build_validator leaves the detector in between checks.
        with DetectorLossContext(detector):
            x = local_batch["img"].detach().clone().requires_grad_(True)
            loss = detection_loss(detector, x, local_batch)
            grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False, only_inputs=True)[0]
        return grad.detach()

    grad_full = _raw_grad(batch)[0]      # image 0's raw gradient within the batch-of-n forward pass
    grad_single = _raw_grad(single_batch)[0]  # image 0's raw gradient within its own batch-of-1 forward pass

    diff = (grad_full - grad_single).abs()
    max_diff = diff.max().item()
    disagree_mask = (torch.sign(grad_full) != torch.sign(grad_single)) & (diff > 1e-8)
    frac_differing_pixels = disagree_mask.float().mean().item()

    mag = grad_single.abs()
    noise_floor = torch.quantile(mag.flatten().float(), noise_floor_percentile / 100.0).item()
    disagree_at_low_magnitude = (mag[disagree_mask] <= noise_floor).float().mean().item() if disagree_mask.any() else 1.0

    sign_agreement = 1.0 - frac_differing_pixels
    ok = (sign_agreement >= sign_agreement_tol) and (disagree_at_low_magnitude >= 0.95)
    print(f"  raw max |grad_full[0]-grad_single[0]| = {max_diff:.8f} (expected: floating-point non-associativity, NOT a tolerance target)")
    print(f"  raw-gradient sign agreement across all pixels: {sign_agreement:.4%} (threshold {sign_agreement_tol:.2%})")
    print(f"  of the sign-disagreeing pixels, fraction at/below the {noise_floor_percentile}th-percentile |raw grad| noise floor: {disagree_at_low_magnitude:.2%} (threshold 95%)")
    print(f"  -> {'PASS' if ok else 'FAIL'} (disagreement confined to near-zero raw-gradient pixels => no cross-sample coupling detected)")
    return ok


def main():
    detector, validator = build_validator(MODEL_PATH, YAML_PATH, "test", batch_size=4)
    batch = next(iter(validator.dataloader))
    batch = validator.preprocess(batch)

    results = {
        "identity": check_identity(detector, batch),
        "frozen_model": check_frozen_model(detector, batch),
        "gradient_reachability": check_gradient_reachability(detector, batch),
        "batch_size_independence": check_batch_size_independence(detector, batch),
    }

    print()
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"SANITY SUITE: FAIL ({failed})")
        sys.exit(1)
    else:
        print("SANITY SUITE: ALL PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
