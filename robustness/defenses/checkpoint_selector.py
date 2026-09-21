"""
Robust checkpoint-selection callback (spec section 13.3):

    "Do not choose best.pt vs last.pt based on test robustness. Implement
    validation-time checkpoint selection using a predeclared criterion."

    S = lambda * mAP_clean + (1-lambda) * mAP_robust

At each validation epoch, in addition to Ultralytics' own clean-fitness
best.pt tracking, this callback runs ONE cheap attack pass over the SAME
validation split (a fixed, documented attack -- e.g. SNUA-GO weak, few
steps, NOT the full 20-step evaluation-time attack from section 23), forms
S, and if it is the best S seen so far, saves the current weights to
`robust_best.pt` in the run's weights directory. The exact rule (lambda,
attack config) is logged to `robust_checkpoint_rule.json` alongside it so
the selection is reproducible and auditable, per spec: "Record the rule in
configuration and results."
"""
import json
from pathlib import Path

import torch

from ..core import BaseAttack
from ..eval_runner import build_validator_for_detector, run_eval_loop


class RobustCheckpointSelector:
    def __init__(self, yaml_path: Path, split: str, attack: BaseAttack, lam: float = 0.5,
                 batch_size: int = 8, workers: int = 0):
        self.yaml_path = Path(yaml_path)
        self.split = split
        self.attack = attack
        self.lam = lam
        self.batch_size = batch_size
        self.workers = workers
        self.best_score = -float("inf")
        self.best_epoch = None
        self.history = []  # per-epoch record for the results file

    def _clean_map_from_trainer(self, trainer) -> float:
        # Ultralytics populates trainer.metrics after each validation epoch
        # with keys like 'metrics/mAP50-95(B)'; fall back to mAP50 if that
        # key is absent in a given ultralytics version.
        m = getattr(trainer, "metrics", {}) or {}
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50(B)"):
            if key in m:
                return float(m[key])
        return float("nan")

    def on_fit_epoch_end(self, trainer):
        detector = trainer.model
        was_training = detector.training
        param_grad_states = [p.requires_grad for p in detector.parameters()]

        clean_map = self._clean_map_from_trainer(trainer)

        validator = build_validator_for_detector(
            detector, self.yaml_path, self.split,
            model_path_str=f"epoch_{trainer.epoch}", device_str=str(trainer.device),
            batch_size=self.batch_size, workers=self.workers,
        )
        stats = run_eval_loop(detector, validator, self.attack, seed=0)
        robust_map = stats.get("metrics/mAP50-95(B)", stats.get("metrics/mAP50(B)", float("nan")))

        score = self.lam * clean_map + (1 - self.lam) * robust_map
        self.history.append({"epoch": int(trainer.epoch), "clean_mAP": clean_map,
                              "robust_mAP": robust_map, "score": score})

        if score > self.best_score:
            self.best_score = score
            self.best_epoch = int(trainer.epoch)
            weights_dir = Path(trainer.wdir)
            weights_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": detector, "epoch": trainer.epoch,
                        "robust_score": score, "clean_mAP": clean_map, "robust_mAP": robust_map},
                       weights_dir / "robust_best.pt")
            print(f"[RobustCheckpointSelector] epoch {trainer.epoch}: NEW best S={score:.4f} "
                  f"(clean={clean_map:.4f}, robust={robust_map:.4f}) -> saved robust_best.pt")
        else:
            print(f"[RobustCheckpointSelector] epoch {trainer.epoch}: S={score:.4f} "
                  f"(clean={clean_map:.4f}, robust={robust_map:.4f}), best so far epoch {self.best_epoch} "
                  f"(S={self.best_score:.4f})")

        # restore training state exactly (run_eval_loop calls detector.eval() and
        # BaseAttack.generate()'s DetectorLossContext already restores requires_grad,
        # but detector.train()/.eval() toggling around the eval loop is this
        # callback's own responsibility since it's outside any attack's context)
        detector.train(was_training)
        for p, g in zip(detector.parameters(), param_grad_states):
            p.requires_grad_(g)

    def save_rule(self, path: Path):
        rule = {
            "criterion": "S = lambda*mAP_clean + (1-lambda)*mAP_robust  (spec section 13.3)",
            "lambda": self.lam,
            "attack": self.attack.name,
            "attack_meta": getattr(self.attack, "__dict__", {}),
            "split": self.split,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "history": self.history,
        }
        with open(path, "w") as f:
            json.dump(rule, f, indent=2, default=str)


def attach_robust_checkpoint_selector(yolo_model, yaml_path: Path, split: str, attack: BaseAttack,
                                       lam: float = 0.5, batch_size: int = 8, workers: int = 0) -> RobustCheckpointSelector:
    """Registers the callback on a `ultralytics.YOLO` model instance (call
    BEFORE `.train(...)`). Returns the selector so its `.save_rule(path)`
    can be called after training finishes."""
    selector = RobustCheckpointSelector(yaml_path, split, attack, lam=lam, batch_size=batch_size, workers=workers)
    yolo_model.add_callback("on_fit_epoch_end", selector.on_fit_epoch_end)
    return selector
