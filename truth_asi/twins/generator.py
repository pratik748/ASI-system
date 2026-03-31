from __future__ import annotations

import random
from typing import Dict, List


class TwinGenerator:
    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _perturb_small(self, value: float) -> float:
        return self._clamp(value + random.uniform(-0.05, 0.05))

    def _perturb_aggressive(self, value: float) -> float:
        return self._clamp(value + random.uniform(-0.10, 0.10))

    def _perturb_inverted(self, value: float, target: float) -> float:
        direction = -1 if target > value else 1
        return self._clamp(value + direction * random.uniform(0.05, 0.10))

    def generate_twins(self, problem_field: Dict, n: int = 50) -> List[Dict[str, Dict[str, float]]]:
        variables = problem_field["variables"]
        target = problem_field["target"]
        keys = list(variables.keys())

        twins: List[Dict[str, Dict[str, float]]] = []
        seen = set()

        while len(twins) < n:
            mode = random.choice(("small", "aggressive", "inverted"))
            new_vars = dict(variables)

            # perturb at least one and up to all variables for diversity
            variable_count = random.randint(1, len(keys))
            perturbed_keys = random.sample(keys, variable_count)

            for key in perturbed_keys:
                if mode == "small":
                    new_vars[key] = self._perturb_small(new_vars[key])
                elif mode == "aggressive":
                    new_vars[key] = self._perturb_aggressive(new_vars[key])
                else:
                    new_vars[key] = self._perturb_inverted(new_vars[key], target[key])

            fingerprint = tuple(round(new_vars[k], 6) for k in keys)
            if fingerprint in seen:
                continue

            seen.add(fingerprint)
            twins.append({"variables": new_vars, "target": dict(target)})

        return twins
