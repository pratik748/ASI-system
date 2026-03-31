from __future__ import annotations

import random
from typing import Dict

from core.tension import TensionCore
from memory.scar import ScarMemory
from simulation.engine import SimulationEngine
from simulation.selector import Selector
from twins.generator import TwinGenerator


def _ask_int(prompt: str, minimum: int = 1) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if value < minimum:
                print(f"Please enter an integer >= {minimum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer.")


def _ask_float_01(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
            if 0.0 <= value <= 1.0:
                return value
            print("Please enter a number between 0 and 1.")
        except ValueError:
            print("Please enter a valid number between 0 and 1.")


def collect_problem_from_user() -> tuple[Dict[str, float], Dict[str, float]]:
    print("Define your optimization problem.")
    variable_count = _ask_int("Number of variables: ", minimum=1)

    current_state: Dict[str, float] = {}
    desired_state: Dict[str, float] = {}

    for idx in range(1, variable_count + 1):
        while True:
            name = input(f"Variable {idx} name: ").strip()
            if not name:
                print("Variable name cannot be empty.")
                continue
            if name in current_state:
                print("Variable name must be unique.")
                continue
            break

        current_state[name] = _ask_float_01(f"Current value for '{name}' (0-1): ")
        desired_state[name] = _ask_float_01(f"Desired value for '{name}' (0-1): ")

    return current_state, desired_state


def run() -> None:
    random.seed()

    current_state, desired_state = collect_problem_from_user()
    max_iterations = _ask_int("Max TRUTH iterations: ", minimum=1)
    max_depth = _ask_int("Max recursive depth: ", minimum=1)
    iteration_budget = _ask_int("Compute budget (nodes) per iteration: ", minimum=1)

    tension = TensionCore(current_state=current_state, desired_state=desired_state)
    twin_gen = TwinGenerator()
    engine = SimulationEngine()
    selector = Selector()
    memory = ScarMemory()

    print("\nStarting interactive TRUTH loop")
    print(f"Initial total tension: {tension.total_tension():.4f}\n")

    for iteration in range(1, max_iterations + 1):
        problem_field = tension.generate_problem_field()
        twins = twin_gen.generate_initial_twins(problem_field, intensity=tension.total_tension())
        twins = [memory.apply_bias(twin) for twin in twins]

        results = []
        consumed = 0

        for twin in twins:
            remaining = iteration_budget - consumed
            if remaining <= 0:
                break

            result = engine.explore_tree(
                twin=twin,
                twin_generator=twin_gen,
                max_depth=max_depth,
                score_gate=0.55,
                budget_left=remaining,
            )
            consumed += int(result["nodes_used"])
            results.append(
                {
                    "twin": {"variables": dict(result["best_state"]), "target": dict(twin["target"])},
                    "score": float(result["final_score"]),
                }
            )

        if not results:
            print(f"Iteration {iteration:02d} | budget exhausted before scoring any twin.")
            break

        ranked = selector.rank(results)
        survivors = selector.eliminate(ranked)

        memory.record_batch(ranked)
        memory.get_bias()

        best = ranked[0]
        best_twin_vars = best["twin"]["variables"]
        for key, value in current_state.items():
            current_state[key] = max(0.0, min(1.0, value + 0.15 * (best_twin_vars[key] - value)))

        tension.current_state = current_state
        tension.compute_tension()

        avg_survivor_score = sum(item["score"] for item in survivors) / len(survivors)
        print(
            f"Iteration {iteration:02d} | "
            f"best score: {best['score']:.4f} | "
            f"avg survivor score: {avg_survivor_score:.4f} | "
            f"total tension: {tension.total_tension():.4f} | "
            f"nodes used: {consumed}"
        )
        print("  Best evolving state:")
        for key in sorted(best_twin_vars):
            print(f"    {key}: {best_twin_vars[key]:.4f}")

        max_possible_score = len(current_state)
        normalized_best = best["score"] / max_possible_score
        if normalized_best > 0.98 or not tension.is_active():
            print(f"\nStopping early at iteration {iteration} (normalized best score={normalized_best:.4f}).")
            break

    print("\nFinal state:")
    for key in sorted(current_state):
        print(f"  {key}: {current_state[key]:.4f}")


if __name__ == "__main__":
    run()
