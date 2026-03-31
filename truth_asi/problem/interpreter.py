from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List


UNIVERSAL_VARIABLES = [
    "risk",
    "opportunity",
    "uncertainty",
    "momentum",
    "resources",
    "efficiency",
]

DOMAIN_VARIABLES: Dict[str, List[str]] = {
    "business": ["revenue", "growth", "cost", "capital", "market"],
    "health": ["energy", "recovery", "stress", "sleep", "focus"],
    "productivity": ["focus", "throughput", "burnout", "consistency"],
    "finance": ["liquidity", "drawdown", "yield", "stability"],
}

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "business": ["business", "customer", "company", "market", "revenue", "growth", "scale"],
    "health": ["health", "fitness", "recovery", "sleep", "nutrition", "stress", "energy"],
    "productivity": ["productivity", "output", "focus", "efficiency", "workflow", "time"],
    "finance": ["capital", "invest", "portfolio", "financial", "risk", "returns", "money"],
}


@dataclass
class InterpretedProblem:
    raw_problem: str
    normalized_problem: str
    intents: List[str]
    domains: List[str]
    variables: List[str]


class ProblemInterpreter:
    def interpret(self, problem: str) -> InterpretedProblem:
        normalized = re.sub(r"\s+", " ", problem.strip().lower())
        words = set(re.findall(r"[a-zA-Z]+", normalized))

        intents: List[str] = []
        for intent_word in ("maximize", "minimize", "optimize", "scale", "safely", "low", "high"):
            if intent_word in words:
                intents.append(intent_word)
        if not intents:
            intents = ["optimize"]

        domain_scores = {
            domain: sum(1 for kw in keywords if kw in normalized)
            for domain, keywords in DOMAIN_KEYWORDS.items()
        }
        ranked_domains = sorted(domain_scores.items(), key=lambda item: item[1], reverse=True)
        domains = [domain for domain, score in ranked_domains if score > 0]

        if not domains:
            domains = ["business" if any(k in normalized for k in ("growth", "scale")) else "productivity"]

        variables = list(dict.fromkeys(UNIVERSAL_VARIABLES))
        for domain in domains:
            for variable in DOMAIN_VARIABLES.get(domain, []):
                if variable not in variables:
                    variables.append(variable)

        return InterpretedProblem(
            raw_problem=problem,
            normalized_problem=normalized,
            intents=intents,
            domains=domains,
            variables=variables,
        )
