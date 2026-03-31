from __future__ import annotations

from typing import Dict

try:
    from ..problem.interpreter import InterpretedProblem
except ImportError:  # pragma: no cover
    from problem.interpreter import InterpretedProblem


NEGATIVE_VARIABLE_HINTS = {
    "risk",
    "uncertainty",
    "cost",
    "stress",
    "burnout",
    "drawdown",
}


class StateConstructor:
    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def build_states(self, interpreted: InterpretedProblem, signals: Dict[str, float]) -> tuple[Dict[str, float], Dict[str, float]]:
        current_state = {name: self._clamp(signals.get(name, 0.5)) for name in interpreted.variables}

        desired_state: Dict[str, float] = {}
        for variable, current in current_state.items():
            if variable in NEGATIVE_VARIABLE_HINTS:
                desired_state[variable] = self._clamp(current * 0.35)
            else:
                desired_state[variable] = self._clamp(0.75 + (1.0 - current) * 0.2)

        if "minimize" in interpreted.intents:
            for name in current_state:
                if name in NEGATIVE_VARIABLE_HINTS:
                    desired_state[name] = 0.1
        if any(intent in interpreted.intents for intent in ("maximize", "scale", "optimize")):
            for name in current_state:
                if name not in NEGATIVE_VARIABLE_HINTS:
                    desired_state[name] = max(desired_state[name], 0.88)

        return current_state, desired_state
