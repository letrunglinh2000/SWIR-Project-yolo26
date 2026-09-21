"""
Unit/constraint/gradient tests for A1 (Random FPN), A2 (SNUA-GO), A3 (Smooth
SNUA) -- spec section 28 "Base transformation tests" and "Gradient tests",
plus the channel-check (18.8) and frozen-model check (18.6) reused from the
PGD sanity suite's pattern.

Run directly: `python robustness/tests/test_snua_unit.py`
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import robustness.sandbox_patch  # noqa: F401

import dataclasses

import torch

from robustness.attacks.random_fpn import RandomFPNAttack
from robustness.attacks.snua import SNUAGoAttack, SmoothSNUAAttack, _tv_smoothness_loss
from robustness.eval_runner import build_validator
from robustness.severities import SNUA_SEVERITIES


def _with_steps(severity, steps):
    return dataclasses.replace(severity, steps=steps)

MODEL_PATH = REPO_ROOT / "runs/train/yolo26n_nslsr_safesplit/weights/best.pt"
YAML_PATH = REPO_ROOT / "dataset_safesplit.yaml"


def _param_hash(detector) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in detector.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_identity(detector, batch):
    print("--- identity check: SNUA-GO(steps=0) leaves image unchanged ---")
    attack = SNUAGoAttack(_with_steps(SNUA_SEVERITIES["medium"], 0))
    result = attack.generate(detector, batch, seed=0)
    ok = torch.equal(result.adv_images, batch["img"])
    print(f"  identical: {ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_constraint(detector, batch):
    print("--- constraint check: |g|<=eps_g, |b|<=eps_b, image in [0,1] ---")
    all_ok = True
    for name, attack in [
        ("random_fpn", RandomFPNAttack.from_severity_name("medium")),
        ("snua_go", SNUAGoAttack.from_severity_name("medium")),
    ]:
        result = attack.generate(detector, batch, seed=0)
        g, b = result.delta_or_params["gain"], result.delta_or_params["offset"]
        sev = SNUA_SEVERITIES["medium"]
        g_ok = (g.abs().max().item() <= sev.eps_gain + 1e-6)
        b_ok = (b.abs().max().item() <= sev.eps_offset + 1e-6)
        img_ok = (result.adv_images.min().item() >= -1e-6) and (result.adv_images.max().item() <= 1 + 1e-6)
        ok = g_ok and b_ok and img_ok
        print(f"  [{name}] |g|<=eps_g: {g_ok}, |b|<=eps_b: {b_ok}, image in [0,1]: {img_ok} -> {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


def test_channel_sharing(detector, batch):
    print("--- channel-sharing check: perturbation identical across all 3 channels ---")
    attack = SNUAGoAttack.from_severity_name("medium")
    result = attack.generate(detector, batch, seed=0)
    delta = result.adv_images - batch["img"]
    same_01 = torch.allclose(delta[:, 0], delta[:, 1], atol=1e-6)
    same_12 = torch.allclose(delta[:, 1], delta[:, 2], atol=1e-6)
    ok = same_01 and same_12
    print(f"  channel0==channel1: {same_01}, channel1==channel2: {same_12} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_smoothness_zero_for_constant(device):
    print("--- smoothness-loss check: TV penalty is zero for a constant parameter vector ---")
    g = torch.full((2, 1, 1, 64), 0.03, device=device)
    b = torch.full((2, 1, 1, 64), -0.01, device=device)
    loss = _tv_smoothness_loss(g, b).item()
    ok = abs(loss) < 1e-8
    print(f"  TV(constant g,b) = {loss:.8f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_gradient_reachability(detector, batch):
    print("--- gradient check: grad reaches g,b; none reach model weights ---")
    for p in detector.parameters():
        p.grad = None
    attack = SNUAGoAttack(_with_steps(SNUA_SEVERITIES["medium"], 1), random_start=False)
    result = attack.generate(detector, batch, seed=0)
    g, b = result.delta_or_params["gain"], result.delta_or_params["offset"]
    finite = torch.isfinite(g).all().item() and torch.isfinite(b).all().item()
    nonzero = (g.abs().sum().item() > 0) or (b.abs().sum().item() > 0)
    no_weight_grad = all(p.grad is None for p in detector.parameters())
    ok = finite and nonzero and no_weight_grad
    print(f"  finite: {finite}, nonzero: {nonzero}, no model-weight grad: {no_weight_grad} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_frozen_model(detector, batch):
    print("--- frozen-model check: SNUA-GO(20 steps) does not alter model weights ---")
    h_before = _param_hash(detector)
    SNUAGoAttack.from_severity_name("medium").generate(detector, batch, seed=0)
    h_after = _param_hash(detector)
    ok = (h_before == h_after)
    print(f"  param hash unchanged: {ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_smooth_snua_runs(detector, batch):
    print("--- Smooth SNUA smoke test: runs, penalizes roughness (not rewards) ---")
    attack = SmoothSNUAAttack.from_severity_name("medium", lambda_smooth=1.0)
    result = attack.generate(detector, batch, seed=0)
    finite = torch.isfinite(result.adv_images).all().item()
    has_meta = ("final_det_loss" in result.meta) and ("final_smooth_loss" in result.meta)
    ok = finite and has_meta
    print(f"  finite output: {finite}, meta logged (det_loss={result.meta.get('final_det_loss')}, "
          f"smooth_loss={result.meta.get('final_smooth_loss')}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    detector, validator = build_validator(MODEL_PATH, YAML_PATH, "test", batch_size=4)
    batch = next(iter(validator.dataloader))
    batch = validator.preprocess(batch)

    results = {
        "identity": test_identity(detector, batch),
        "constraint": test_constraint(detector, batch),
        "channel_sharing": test_channel_sharing(detector, batch),
        "smoothness_zero_for_constant": test_smoothness_zero_for_constant(batch["img"].device),
        "gradient_reachability": test_gradient_reachability(detector, batch),
        "frozen_model": test_frozen_model(detector, batch),
        "smooth_snua_smoke": test_smooth_snua_runs(detector, batch),
    }

    print()
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"SNUA UNIT SUITE: FAIL ({failed})")
        sys.exit(1)
    print("SNUA UNIT SUITE: ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
