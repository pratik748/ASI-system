from __future__ import annotations

import random
from statistics import mean, variance
from typing import Dict, List

from twins.generator import TwinGenerator


class SimulationEngine:
    def __init__(self, move_rate: float = 0.1, noise: float = 0.02, variance_penalty: float = 0.08) -> None:
        self.move_rate = move_rate
        self.noise = noise
        self.variance_penalty = variance_penalty

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def run_simulation(self, twin: Dict[str, Dict[str, float]], steps: int = 12) -> List[Dict[str, float]]:
        current = dict(twin["variables"])
        target = twin["target"]
        trajectory: List[Dict[str, float]] = [dict(current)]

        for _ in range(steps):
            for key, value in current.items():
                drift = (target[key] - value) * self.move_rate
                noise = random.uniform(-self.noise, self.noise)
                current[key] = self._clamp(value + drift + noise)
            trajectory.append(dict(current))

        return trajectory

    def _tradeoff_penalty(self, state: Dict[str, float]) -> float:
        penalty = 0.0
        if "growth" in state and "risk" in state:
            penalty += max(0.0, state["growth"] - (1.0 - state["risk"])) * 0.25
        if "opportunity" in state and "resources" in state:
            penalty += max(0.0, state["opportunity"] - state["resources"]) * 0.15
        if "productivity" in state and "stress" in state:
            penalty += max(0.0, state["stress"] - 0.6) * 0.2
        return penalty

    def evaluate_outcome(self, trajectory: List[Dict[str, float]], target: Dict[str, float]) -> float:
        final_state = trajectory[-1]
        long_term_state = {
            key: mean([snapshot[key] for snapshot in trajectory[max(1, len(trajectory) // 2):]])
            for key in final_state
        }

        short_term_score = sum(1.0 - abs(target[k] - final_state[k]) for k in final_state)
        long_term_score = sum(1.0 - abs(target[k] - long_term_state[k]) for k in long_term_state)

        flat_values = [value for state in trajectory for value in state.values()]
        instability = variance(flat_values) if len(flat_values) > 1 else 0.0

        score = (short_term_score * 0.45) + (long_term_score * 0.55)
        score -= instability * self.variance_penalty
        score -= self._tradeoff_penalty(final_state)
        return score

    def _dynamic_branch_factor(self, normalized_score: float, depth: int, max_depth: int) -> int:
        if normalized_score < 0.30:
            return 0

        depth_decay = max(0.2, 1.0 - (depth / max_depth))
        if normalized_score < 0.55:
            return max(1, int(2 * depth_decay))
        if normalized_score < 0.80:
            return max(1, int(4 * depth_decay))
        return max(1, int(6 * depth_decay))

    def explore_tree(
        self,
        twin: Dict[str, Dict[str, float]],
        twin_generator: TwinGenerator,
        max_depth: int,
        score_gate: float,
        budget_left: int,
        depth: int = 0,
    ) -> Dict[str, float | Dict[str, float] | int]:
        trajectory = self.run_simulation(twin)
        base_score = self.evaluate_outcome(trajectory, twin["target"])

        best_score = base_score
        best_state = dict(trajectory[-1])
        nodes_used = 1
        max_possible = len(best_state)
        normalized = best_score / max_possible

        if budget_left <= 1 or depth >= max_depth:
            return {"final_score": best_score, "best_state": best_state, "nodes_used": nodes_used}

        branch_factor = self._dynamic_branch_factor(normalized, depth, max_depth)
        if normalized < score_gate:
            branch_factor = min(branch_factor, 1)

        if branch_factor <= 0:
            return {"final_score": best_score, "best_state": best_state, "nodes_used": nodes_used}

        mutation_scale = max(0.01, 0.08 * (1.0 - normalized))
        children = twin_generator.spawn_children(
            {"variables": best_state, "target": dict(twin["target"])} ,
            branch_factor=branch_factor,
            mutation_scale=mutation_scale,
        )

        for child in children:
            remaining = budget_left - nodes_used
            if remaining <= 0:
                break

            child_result = self.explore_tree(
                twin=child,
                twin_generator=twin_generator,
                max_depth=max_depth,
                score_gate=score_gate,
                budget_left=remaining,
                depth=depth + 1,
            )
            nodes_used += int(child_result["nodes_used"])

            if float(child_result["final_score"]) > best_score:
                best_score = float(child_result["final_score"])
                best_state = dict(child_result["best_state"])

        return {"final_score": best_score, "best_state": best_state, "nodes_used": nodes_used}
