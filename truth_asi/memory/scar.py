from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


class ScarMemory:
    def __init__(self) -> None:
        self.records: List[Dict] = []
        self.bias: Dict[str, str] = {}

    def record(self, twin: Dict, score: float, record_type: str) -> None:
        self.records.append(
            {
                "variables": dict(twin["variables"]),
                "score": score,
                "type": record_type,
            }
        )

    def record_batch(self, ranked_results: List[Dict]) -> None:
        if not ranked_results:
            return

        total = len(ranked_results)
        success_cutoff = max(1, int(total * 0.20))
        failure_cutoff = max(1, int(total * 0.30))

        for idx, result in enumerate(ranked_results):
            if idx < success_cutoff:
                rtype = "success"
            elif idx >= total - failure_cutoff:
                rtype = "failure"
            else:
                continue
            self.record(result["twin"], result["score"], rtype)

    def get_bias(self) -> Dict[str, str]:
        success = [r for r in self.records if r["type"] == "success"]
        failure = [r for r in self.records if r["type"] == "failure"]

        if not success or not failure:
            self.bias = {}
            return self.bias

        success_avgs = defaultdict(float)
        failure_avgs = defaultdict(float)

        keys = success[0]["variables"].keys()
        for key in keys:
            success_avgs[key] = sum(item["variables"][key] for item in success) / len(success)
            failure_avgs[key] = sum(item["variables"][key] for item in failure) / len(failure)

        self.bias = {
            key: "increase" if success_avgs[key] >= failure_avgs[key] else "decrease"
            for key in keys
        }
        return self.bias

    def apply_bias(self, twin: Dict[str, Dict[str, float]], strength: float = 0.01) -> Dict[str, Dict[str, float]]:
        if not self.bias:
            return twin

        adjusted = {
            "variables": dict(twin["variables"]),
            "target": dict(twin["target"]),
        }

        for key, direction in self.bias.items():
            delta = strength if direction == "increase" else -strength
            adjusted["variables"][key] = max(0.0, min(1.0, adjusted["variables"][key] + delta))

        return adjusted
