from __future__ import annotations

import random
from statistics import variance
from typing import Dict, List


class SimulationEngine:
    def __init__(self, move_rate: float = 0.1, noise: float = 0.02, variance_penalty: float = 0.05) -> None:
        self.move_rate = move_rate
        self.noise = noise
        self.variance_penalty = variance_penalty

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def run_simulation(self, twin: Dict[str, Dict[str, float]], steps: int = 10) -> List[Dict[str, float]]:
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

    def evaluate_outcome(self, trajectory: List[Dict[str, float]], target: Dict[str, float]) -> float:
        final_state = trajectory[-1]
        score = sum(1.0 - abs(target[k] - final_state[k]) for k in final_state)

        flat_values = [value for state in trajectory for value in state.values()]
        traj_variance = variance(flat_values) if len(flat_values) > 1 else 0.0
        score -= traj_variance * self.variance_penalty
        return score

    def simulate_recursive(self, twin: Dict[str, Dict[str, float]], depth: int = 2) -> Dict[str, float]:
        trajectory = self.run_simulation(twin)
        base_score = self.evaluate_outcome(trajectory, twin["target"])

        if depth <= 0:
            return {"final_score": base_score}

        final_state = trajectory[-1]
        child_scores = []
        child_count = random.randint(2, 3)

        for _ in range(child_count):
            child_variables = {
                k: self._clamp(v + random.uniform(-0.03, 0.03))
                for k, v in final_state.items()
            }
            child_twin = {
                "variables": child_variables,
                "target": dict(twin["target"]),
            }
            child_scores.append(self.simulate_recursive(child_twin, depth=depth - 1)["final_score"])

        final_score = 0.5 * base_score + 0.5 * (sum(child_scores) / len(child_scores))
        return {"final_score": final_score}
