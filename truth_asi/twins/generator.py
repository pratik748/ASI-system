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
        return self._clamp(value + random.uniform(-0.12, 0.12))

    def _perturb_inverted(self, value: float, target: float) -> float:
        direction = -1 if target > value else 1
        return self._clamp(value + direction * random.uniform(0.05, 0.12))

    def generate_initial_twins(
        self,
        problem_field: Dict,
        intensity: float,
        budget: int,
    ) -> List[Dict[str, Dict[str, float]]]:
        variables = problem_field["variables"]
        target = problem_field["target"]
        keys = list(variables.keys())

        base = max(4, len(keys))
        budget_factor = max(1, min(12, budget // 30))
        twin_count = max(4, min(base * budget_factor, base + int(intensity * 10)))

        twins: List[Dict[str, Dict[str, float]]] = []
        seen = set()

        while len(twins) < twin_count:
            mode = random.choice(("small", "aggressive", "inverted"))
            new_vars = dict(variables)

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

    def spawn_children(
        self,
        parent: Dict[str, Dict[str, float]],
        branch_factor: int,
        mutation_scale: float,
    ) -> List[Dict[str, Dict[str, float]]]:
        children: List[Dict[str, Dict[str, float]]] = []
        parent_vars = parent["variables"]
        target = parent["target"]

        keys = list(parent_vars.keys())
        for _ in range(branch_factor):
            child_vars = dict(parent_vars)
            mutation_count = random.randint(1, len(keys))
            mutate_keys = random.sample(keys, mutation_count)
            for key in mutate_keys:
                toward_target = (target[key] - child_vars[key]) * random.uniform(0.2, 1.1)
                noise = random.uniform(-mutation_scale, mutation_scale)
                child_vars[key] = self._clamp(child_vars[key] + toward_target + noise)

            children.append({"variables": child_vars, "target": dict(target)})

        return children
