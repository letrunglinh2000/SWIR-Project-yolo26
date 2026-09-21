"""
Attack scheduler for robust training (spec section 21, D2/D3/D5).

Per-BATCH sampling from a named set of attack factories (or "clean" for no
attack), weighted by a probability table -- matches spec D5's suggested
starting scheduler literally:

    p(clean) = 0.25
    p(SNUA)  = 0.25
    p(ARIA)  = 0.25
    p(SNUA+ARIA) = 0.25

("Treat these as starting values, not final hyperparameters" -- weights are
a constructor argument, not hardcoded.)

D2 (random augmentation) and D3 (single-attack AT) both reduce to a
2-entry scheduler ({"clean": 1-adv_ratio, "<attack>": adv_ratio}), so this
one class covers D2/D3/D5 -- no separate code path per defense.
"""
import random
from typing import Callable, Dict, Optional

import torch


class AttackScheduler:
    """`weights` maps a name -> probability (renormalized if they don't sum
    to 1). `factories` maps the SAME names (except "clean", which is
    implicit and needs no factory) -> a zero-arg callable returning a fresh
    `BaseAttack` instance for that name. Sampling is per-batch (the whole
    batch gets the same attack), matching spec section 21's "sample or
    schedule attack families" wording literally.
    """

    def __init__(self, weights: Dict[str, float], factories: Dict[str, Callable[[], object]]):
        for name in weights:
            if name != "clean" and name not in factories:
                raise ValueError(f"no factory registered for scheduler entry '{name}'")
        total = sum(weights.values())
        self.names = list(weights.keys())
        self.probs = [w / total for w in weights.values()]
        self.factories = factories
        self.counts = {name: 0 for name in self.names}  # running mix, for logging

    def sample(self, rng: Optional[random.Random] = None) -> Optional[object]:
        """Returns a fresh attack instance, or None for 'clean' (no attack)."""
        r = rng or random
        name = r.choices(self.names, weights=self.probs, k=1)[0]
        self.counts[name] += 1
        if name == "clean":
            return None
        return self.factories[name]()

    def mix_summary(self) -> Dict[str, float]:
        total = sum(self.counts.values())
        if total == 0:
            return {name: 0.0 for name in self.names}
        return {name: c / total for name, c in self.counts.items()}
