"""
SWIRRobustDetectionTrainer (spec section 22): minimally-invasive Ultralytics
DetectionTrainer subclass that injects SWIR-specific (and/or generic PGD)
adversarial examples into each training batch, reusing this project's
existing `robustness/attacks/*` implementations rather than re-deriving the
freeze/generate/unfreeze pattern (train_attack.py's PGDDetectionTrainer did
that ad hoc for PGD only; every attack here already implements it once via
`BaseAttack.generate()` -> `DetectorLossContext`).

Nested-optimization safety (spec section 22's explicit sequence):
    freeze theta for attack generation   <- DetectorLossContext.__enter__
    generate I_adv with current theta    <- attack.generate(...)
    unfreeze theta                       <- DetectorLossContext.__exit__
    zero model optimizer grads           <- Ultralytics' own train loop
    compute training loss on I_adv       <- Ultralytics' own train loop
    backprop into theta / optimizer.step <- Ultralytics' own train loop

This module only overrides `preprocess_batch` (called once per training
step, BEFORE the trainer's own forward/backward) to substitute a per-batch
mix of clean/adversarial images. It never touches the trainer's own
forward/backward/optimizer code -- that stays exactly Ultralytics'.

First implementation uses DETACHED adversarial examples (spec: "Do not
retain the full inner attack computation graph for the outer update unless
intentionally implementing higher-order adversarial training") -- every
`BaseAttack.generate()` implementation already returns `.detach()`-ed
tensors, so no extra detach is needed here.
"""
from typing import Optional

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from ..core import prepare_detector_for_attack


class SWIRRobustDetectionTrainer(DetectionTrainer):
    """Class-level config (set on the class BEFORE calling `YOLO.train()`,
    matching this project's existing `PGDDetectionTrainer` convention):

        SWIRRobustDetectionTrainer.attack_scheduler = <AttackScheduler instance>
        SWIRRobustDetectionTrainer.adv_ratio = 1.0   # scheduler already
            encodes the clean/attack mix via its "clean" entry weight, so
            adv_ratio is normally 1.0 (always consult the scheduler) --
            kept as a knob for a global "eval sanity" override to 0.0.
        SWIRRobustDetectionTrainer.adv_start_epoch = 0

    NOTE: named `attack_scheduler`, NOT `scheduler` -- `BaseTrainer` (the
    Ultralytics superclass) already owns `self.scheduler` for its LR
    scheduler (a `torch.optim.lr_scheduler.LambdaLR`), set up internally
    during `_setup_train`. A same-named class attribute here would be
    silently shadowed/overwritten by that assignment (confirmed: an earlier
    version of this file used `scheduler` and crashed with
    `AttributeError: 'LambdaLR' object has no attribute 'sample'` the
    moment `preprocess_batch` ran, because by then Ultralytics had already
    replaced it with the real LR scheduler).
    """

    attack_scheduler = None  # AttackScheduler; must be set before training starts
    adv_ratio = 1.0
    adv_start_epoch = 0
    log_mix_every_n_batches = 200

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        if self.attack_scheduler is None:
            raise RuntimeError(
                "SWIRRobustDetectionTrainer.attack_scheduler must be set to an AttackScheduler "
                "instance before calling YOLO.train() with this trainer."
            )
        if self.epoch < self.adv_start_epoch or self.adv_ratio <= 0.0:
            return batch

        clean_img = batch["img"].detach()
        attack_model = unwrap_model(self.model)
        prepare_detector_for_attack(attack_model)

        attack = self.attack_scheduler.sample()
        if attack is None:  # sampled "clean" -- no attack this batch
            return batch

        result = attack.generate(attack_model, batch, seed=None)
        adv_img = result.adv_images

        if self.adv_ratio >= 1.0:
            batch["img"] = adv_img
        else:
            batch_size = clean_img.shape[0]
            use_adv = (torch.rand(batch_size, device=clean_img.device) < self.adv_ratio).view(batch_size, 1, 1, 1)
            batch["img"] = torch.where(use_adv, adv_img, clean_img)

        if hasattr(self, "_swir_batch_count"):
            self._swir_batch_count += 1
        else:
            self._swir_batch_count = 1
        if self._swir_batch_count % self.log_mix_every_n_batches == 0:
            print(f"[SWIRRobustDetectionTrainer] batch {self._swir_batch_count}: "
                  f"attack '{attack.name}', running mix = {self.attack_scheduler.mix_summary()}")

        return batch
