from __future__ import annotations

import math
import re
from typing import Dict, Iterable

import requests

from problem.interpreter import InterpretedProblem


class InternetSignalFetcher:
    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _fetch_duckduckgo_text(self, query: str) -> str:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        parts = [payload.get("AbstractText", "")]
        for item in payload.get("RelatedTopics", [])[:5]:
            if isinstance(item, dict):
                text = item.get("Text")
                if text:
                    parts.append(text)
        return " ".join(part for part in parts if part)

    def _fetch_wikipedia_summary(self, topic: str) -> str:
        safe_topic = re.sub(r"\s+", "_", topic.strip())
        response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_topic}",
            timeout=self.timeout,
            headers={"accept": "application/json"},
        )
        if response.status_code >= 400:
            return ""
        payload = response.json()
        return str(payload.get("extract", ""))

    def _aggregate_text(self, interpreted: InterpretedProblem) -> str:
        query = f"{interpreted.raw_problem} trends risks opportunities"
        chunks = []
        try:
            chunks.append(self._fetch_duckduckgo_text(query))
        except requests.RequestException:
            chunks.append("")

        for domain in interpreted.domains[:2]:
            try:
                chunks.append(self._fetch_wikipedia_summary(domain))
            except requests.RequestException:
                continue

        return " ".join(chunk for chunk in chunks if chunk)

    def _keyword_density(self, text: str, words: Iterable[str]) -> float:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        if not tokens:
            return 0.5

        total = len(tokens)
        score = sum(tokens.count(word) for word in words)
        ratio = score / max(1, math.sqrt(total))
        return self._clamp(ratio)

    def fetch_signals(self, interpreted: InterpretedProblem) -> Dict[str, float]:
        text = self._aggregate_text(interpreted)

        if not text:
            return {variable: 0.5 for variable in interpreted.variables}

        keyword_map = {
            "risk": ["risk", "loss", "downturn", "threat", "volatility"],
            "opportunity": ["opportunity", "growth", "upside", "expansion"],
            "uncertainty": ["uncertain", "volatility", "unknown", "change"],
            "momentum": ["momentum", "trend", "increase", "accelerate"],
            "resources": ["resource", "capital", "cash", "support"],
            "efficiency": ["efficient", "productivity", "optimize", "output"],
            "revenue": ["revenue", "income", "sales"],
            "growth": ["growth", "expand", "scale"],
            "cost": ["cost", "expense", "burn", "overhead"],
            "capital": ["capital", "funding", "cash", "liquidity"],
            "market": ["market", "demand", "competition"],
            "energy": ["energy", "vitality", "exercise"],
            "recovery": ["recovery", "rest", "sleep"],
            "stress": ["stress", "fatigue", "anxiety"],
            "sleep": ["sleep", "rest"],
            "focus": ["focus", "attention", "concentration"],
            "throughput": ["throughput", "output", "delivery"],
            "burnout": ["burnout", "overload", "strain"],
            "consistency": ["consistent", "routine", "habit"],
            "liquidity": ["liquidity", "cash", "reserves"],
            "drawdown": ["drawdown", "decline", "loss"],
            "yield": ["yield", "return", "gain"],
            "stability": ["stability", "resilient", "steady"],
        }

        return {
            variable: self._keyword_density(text, keyword_map.get(variable, [variable]))
            for variable in interpreted.variables
        }
