from __future__ import annotations

import random

from core.tension import TensionCore
from memory.scar import ScarMemory
from simulation.engine import SimulationEngine
from simulation.selector import Selector
from twins.generator import TwinGenerator


def run(iterations: int = 15, variable_count: int = 8) -> None:
    random.seed(42)

    current_state = {f"v{i}": random.random() for i in range(1, variable_count + 1)}
    desired_state = {key: 1.0 for key in current_state}

    tension = TensionCore(current_state=current_state, desired_state=desired_state)
    twin_gen = TwinGenerator()
    engine = SimulationEngine()
    selector = Selector()
    memory = ScarMemory()

    print("Starting truth_asi loop")
    print(f"Initial total tension: {tension.total_tension():.4f}\n")

    for iteration in range(1, iterations + 1):
        problem_field = tension.generate_problem_field()
        twins = twin_gen.generate_twins(problem_field, n=50)
        twins = [memory.apply_bias(twin) for twin in twins]

        results = []
        for twin in twins:
            score = engine.simulate_recursive(twin, depth=2)["final_score"]
            results.append({"twin": twin, "score": score})

        ranked = selector.rank(results)
        survivors = selector.eliminate(ranked)

        memory.record_batch(ranked)
        memory.get_bias()

        best = ranked[0]
        best_twin_vars = best["twin"]["variables"]
        for key, value in current_state.items():
            current_state[key] = max(0.0, min(1.0, value + 0.1 * (best_twin_vars[key] - value)))

        tension.current_state = current_state

        avg_survivor_score = sum(item["score"] for item in survivors) / len(survivors)
        print(
            f"Iteration {iteration:02d} | "
            f"best score: {best['score']:.4f} | "
            f"avg survivor score: {avg_survivor_score:.4f} | "
            f"total tension: {tension.total_tension():.4f}"
        )

        max_possible_score = len(current_state)
        normalized_best = best["score"] / max_possible_score
        if normalized_best > 0.95 or not tension.is_active():
            print(f"\nStopping early at iteration {iteration} (normalized best score={normalized_best:.4f}).")
            break

    print("\nFinal state:")
    for key in sorted(current_state):
        print(f"  {key}: {current_state[key]:.4f}")


if __name__ == "__main__":
    run()
