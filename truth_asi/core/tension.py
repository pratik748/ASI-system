from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TensionCore:
    current_state: Dict[str, float]
    desired_state: Dict[str, float]
    tension_map: Dict[str, float] = field(default_factory=dict)

    def compute_tension(self) -> Dict[str, float]:
        self.tension_map = {
            key: abs(self.desired_state[key] - self.current_state[key])
            for key in self.current_state
        }
        return self.tension_map

    def total_tension(self) -> float:
        if not self.tension_map:
            self.compute_tension()
        return sum(self.tension_map.values())

    def is_active(self, threshold: float = 0.01) -> bool:
        return self.total_tension() > threshold

    def generate_problem_field(self) -> Dict[str, Dict[str, float] | float]:
        return {
            "variables": dict(self.current_state),
            "target": dict(self.desired_state),
            "tension": self.compute_tension(),
            "intensity": self.total_tension(),
        }
