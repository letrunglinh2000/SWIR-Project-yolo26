"""
Unit/constraint/gradient tests for A7 (Combined SNUA+ARIA, joint mode).

Run directly: `python robustness/tests/test_combined_unit.py`
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401

import torch

from robustness.attacks.combined import CombinedSNUAARIAAttack
from robustness.eval_runner import build_validator
from robustness.severities import ARIA_SEVERITIES, SNUA_SEVERITIES

MODEL_PATH = REPO_ROOT / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt"
YAML_PATH = REPO_ROOT / "dataset_safesplit.yaml"


def _param_hash(detector) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in detector.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_constraint(detector, batch):
    print("--- constraint check: combined |g|<=eps_g, |b|<=eps_b, |A-1|<=eps_field, image in [0,1] ---")
    attack = CombinedSNUAARIAAttack.from_severity_names("medium", "medium")
    result = attack.generate(detector, batch, seed=0)
    g, b, A = result.delta_or_params["gain"], result.delta_or_params["offset"], result.delta_or_params["field_upsampled"]
    sev_snua, sev_aria = SNUA_SEVERITIES["medium"], ARIA_SEVERITIES["medium"]
    g_ok = (g.abs().max().item() <= sev_snua.eps_gain + 1e-6)
    b_ok = (b.abs().max().item() <= sev_snua.eps_offset + 1e-6)
    a_ok = ((A - 1.0).abs().max().item() <= sev_aria.eps_field + 1e-6)
    img_ok = (result.adv_images.min().item() >= -1e-6) and (result.adv_images.max().item() <= 1 + 1e-6)
    ok = g_ok and b_ok and a_ok and img_ok
    print(f"  |g|<=eps_g: {g_ok}, |b|<=eps_b: {b_ok}, |A-1|<=eps_field: {a_ok}, image in [0,1]: {img_ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_gradient_reachability(detector, batch):
    print("--- gradient check: grad reaches g,b,z jointly; none reach model weights ---")
    for p in detector.parameters():
        p.grad = None
    attack = CombinedSNUAARIAAttack.from_severity_names("medium", "medium", random_start=False)
    result = attack.generate(detector, batch, seed=0)
    g, b = result.delta_or_params["gain"], result.delta_or_params["offset"]
    finite = torch.isfinite(g).all().item() and torch.isfinite(b).all().item()
    nonzero = (g.abs().sum().item() > 0) and (b.abs().sum().item() > 0)
    no_weight_grad = all(p.grad is None for p in detector.parameters())
    ok = finite and nonzero and no_weight_grad
    print(f"  finite: {finite}, nonzero: {nonzero}, no model-weight grad: {no_weight_grad} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_frozen_model(detector, batch):
    print("--- frozen-model check: combined attack does not alter model weights ---")
    h_before = _param_hash(detector)
    CombinedSNUAARIAAttack.from_severity_names("medium", "medium").generate(detector, batch, seed=0)
    h_after = _param_hash(detector)
    ok = (h_before == h_after)
    print(f"  param hash unchanged: {ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_stronger_than_either_component(detector, batch):
    """Sanity: joint combined attack should generally not be WEAKER than
    either of its two components alone at the same severity. This is this
    implementation's OWN sanity check, not a spec requirement -- spec section
    4.8 defines only the formula, constraints, and joint/alternating
    optimization modes, with no claim about combined-vs-component strength."""
    print("--- component-dominance sanity: combined loss >= max(SNUA-only, ARIA-only) loss (same init) ---")
    from robustness.attacks.snua import SNUAGoAttack
    from robustness.attacks.aria import ARIAAttack
    from robustness.core import detection_loss

    combined = CombinedSNUAARIAAttack.from_severity_names("medium", "medium", random_start=False)
    snua_only = SNUAGoAttack.from_severity_name("medium", random_start=False)
    aria_only = ARIAAttack.from_severity_name("medium", random_start=False)

    r_combined = combined.generate(detector, batch, seed=0)
    r_snua = snua_only.generate(detector, batch, seed=0)
    r_aria = aria_only.generate(detector, batch, seed=0)

    from robustness.core import DetectorLossContext
    with DetectorLossContext(detector):
        loss_combined = detection_loss(detector, r_combined.adv_images, batch).item()
        loss_snua = detection_loss(detector, r_snua.adv_images, batch).item()
        loss_aria = detection_loss(detector, r_aria.adv_images, batch).item()

    ok = loss_combined >= max(loss_snua, loss_aria) - 1e-3  # small tolerance for optimization variance
    print(f"  loss_combined={loss_combined:.4f}, loss_snua_only={loss_snua:.4f}, loss_aria_only={loss_aria:.4f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    detector, validator = build_validator(MODEL_PATH, YAML_PATH, "test", batch_size=4)
    batch = next(iter(validator.dataloader))
    batch = validator.preprocess(batch)

    results = {
        "constraint": test_constraint(detector, batch),
        "gradient_reachability": test_gradient_reachability(detector, batch),
        "frozen_model": test_frozen_model(detector, batch),
        "component_dominance": test_stronger_than_either_component(detector, batch),
    }

    print()
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"COMBINED UNIT SUITE: FAIL ({failed})")
        sys.exit(1)
    print("COMBINED UNIT SUITE: ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
