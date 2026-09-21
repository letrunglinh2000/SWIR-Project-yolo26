"""
Unit/constraint/gradient tests for A4 (Global radiometric), A5 (Random
smooth radiometric control), A6 (ARIA) -- mirrors test_snua_unit.py's
structure/coverage for the radiometric attack family.

Run directly: `python robustness/tests/test_aria_unit.py`
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401

import torch

from robustness.attacks.aria import ARIAAttack
from robustness.attacks.radiometric import GlobalRadiometricAttack, RandomSmoothRadiometricAttack
from robustness.eval_runner import build_validator
from robustness.severities import ARIA_SEVERITIES, RADIOMETRIC_SEVERITIES

MODEL_PATH = REPO_ROOT / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt"
YAML_PATH = REPO_ROOT / "dataset_safesplit.yaml"


def _param_hash(detector) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in detector.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_identity_global(detector, batch):
    print("--- identity check: GlobalRadiometric(steps=0, random_start=False) leaves image unchanged ---")
    attack = GlobalRadiometricAttack.from_severity_name("medium", steps=0, random_start=False)
    result = attack.generate(detector, batch, seed=0)
    ok = torch.equal(result.adv_images, batch["img"])
    print(f"  identical: {ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_identity_aria(detector, batch):
    print("--- identity check: ARIA(steps=0) leaves image unchanged ---")
    from robustness.severities import ARIASeverity
    attack = ARIAAttack(ARIASeverity(eps_field=ARIA_SEVERITIES["medium"].eps_field, step_field=ARIA_SEVERITIES["medium"].step_field, steps=0))
    result = attack.generate(detector, batch, seed=0)
    ok = torch.equal(result.adv_images, batch["img"])
    print(f"  identical: {ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_constraint_global(detector, batch):
    print("--- constraint check: GlobalRadiometric |alpha-1|<=eps_alpha, |beta|<=eps_beta ---")
    sev = RADIOMETRIC_SEVERITIES["medium"]
    attack = GlobalRadiometricAttack.from_severity_name("medium")
    result = attack.generate(detector, batch, seed=0)
    alpha, beta = result.delta_or_params["alpha"], result.delta_or_params["beta"]
    a_ok = ((alpha - 1.0).abs().max().item() <= sev["eps_alpha"] + 1e-6)
    b_ok = (beta.abs().max().item() <= sev["eps_beta"] + 1e-6)
    img_ok = (result.adv_images.min().item() >= -1e-6) and (result.adv_images.max().item() <= 1 + 1e-6)
    ok = a_ok and b_ok and img_ok
    print(f"  |alpha-1|<=eps_alpha: {a_ok}, |beta|<=eps_beta: {b_ok}, image in [0,1]: {img_ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_constraint_aria(detector, batch):
    print("--- constraint check: ARIA / RandomSmoothRadiometric |A-1|<=eps_field ---")
    all_ok = True
    for name, attack in [
        ("random_radiometric", RandomSmoothRadiometricAttack.from_severity_name("medium")),
        ("aria", ARIAAttack.from_severity_name("medium")),
    ]:
        result = attack.generate(detector, batch, seed=0)
        eps_field = ARIA_SEVERITIES["medium"].eps_field
        A = result.delta_or_params["field_upsampled"]
        field_ok = ((A - 1.0).abs().max().item() <= eps_field + 1e-6)
        img_ok = (result.adv_images.min().item() >= -1e-6) and (result.adv_images.max().item() <= 1 + 1e-6)
        ok = field_ok and img_ok
        print(f"  [{name}] |A-1|<=eps_field: {field_ok}, image in [0,1]: {img_ok} -> {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


def test_channel_sharing(detector, batch):
    print("--- channel-sharing check: ARIA field identical across all 3 channels ---")
    attack = ARIAAttack.from_severity_name("medium")
    result = attack.generate(detector, batch, seed=0)
    delta = result.adv_images - batch["img"]
    same_01 = torch.allclose(delta[:, 0], delta[:, 1], atol=1e-6)
    same_12 = torch.allclose(delta[:, 1], delta[:, 2], atol=1e-6)
    ok = same_01 and same_12
    print(f"  channel0==channel1: {same_01}, channel1==channel2: {same_12} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_gradient_reachability(detector, batch):
    print("--- gradient check: grad reaches z (ARIA) / alpha,beta (global); none reach model weights ---")
    all_ok = True
    for name, attack in [
        ("global_radiometric", GlobalRadiometricAttack.from_severity_name("medium", steps=1, random_start=False)),
        ("aria", ARIAAttack.from_severity_name("medium", random_start=False)),
    ]:
        for p in detector.parameters():
            p.grad = None
        result = attack.generate(detector, batch, seed=0)
        params = [v for k, v in result.delta_or_params.items() if k != "field_upsampled"]
        finite = all(torch.isfinite(p).all().item() for p in params)
        nonzero = any(p.abs().sum().item() > 0 for p in params)
        no_weight_grad = all(p.grad is None for p in detector.parameters())
        ok = finite and nonzero and no_weight_grad
        print(f"  [{name}] finite: {finite}, nonzero: {nonzero}, no model-weight grad: {no_weight_grad} -> {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


def test_frozen_model(detector, batch):
    print("--- frozen-model check: ARIA(20 steps) does not alter model weights ---")
    h_before = _param_hash(detector)
    ARIAAttack.from_severity_name("medium").generate(detector, batch, seed=0)
    h_after = _param_hash(detector)
    ok = (h_before == h_after)
    print(f"  param hash unchanged: {ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    detector, validator = build_validator(MODEL_PATH, YAML_PATH, "test", batch_size=4)
    batch = next(iter(validator.dataloader))
    batch = validator.preprocess(batch)

    results = {
        "identity_global": test_identity_global(detector, batch),
        "identity_aria": test_identity_aria(detector, batch),
        "constraint_global": test_constraint_global(detector, batch),
        "constraint_aria": test_constraint_aria(detector, batch),
        "channel_sharing": test_channel_sharing(detector, batch),
        "gradient_reachability": test_gradient_reachability(detector, batch),
        "frozen_model": test_frozen_model(detector, batch),
    }

    print()
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"ARIA UNIT SUITE: FAIL ({failed})")
        sys.exit(1)
    print("ARIA UNIT SUITE: ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
