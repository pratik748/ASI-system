from __future__ import annotations

import random
from typing import Dict, List


class Selector:
    def rank(self, results: List[Dict]) -> List[Dict]:
        return sorted(results, key=lambda item: item["score"], reverse=True)

    def eliminate(self, ranked: List[Dict]) -> List[Dict]:
        if not ranked:
            return []

        total = len(ranked)
        top_count = max(1, int(total * 0.15))
        random_count = max(1, int(total * 0.05))

        top = ranked[:top_count]
        remaining = ranked[top_count:]
        sampled = random.sample(remaining, k=min(random_count, len(remaining))) if remaining else []
        return top + sampled
