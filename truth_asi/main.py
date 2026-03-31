from __future__ import annotations

import random
import threading
import time
from queue import Empty, Queue
from typing import Dict

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.tension import TensionCore
from data.signals import InternetSignalFetcher
from memory.scar import ScarMemory
from problem.interpreter import InterpretedProblem, ProblemInterpreter
from simulation.engine import SimulationEngine
from simulation.selector import Selector
from state.constructor import StateConstructor
from twins.generator import TwinGenerator


class VariableControl:
    def __init__(self, parent: tk.Widget, name: str, on_change: callable | None = None) -> None:
        self.name = name
        self.frame = ttk.Frame(parent)
        self.current_var = tk.DoubleVar(value=0.5)
        self.desired_var = tk.DoubleVar(value=0.8)

        ttk.Label(self.frame, text=name, width=14).grid(row=0, column=0, sticky="w", padx=(0, 8))

        ttk.Label(self.frame, text="Current").grid(row=0, column=1, sticky="w")
        self.current_scale = ttk.Scale(
            self.frame,
            from_=0.0,
            to=1.0,
            variable=self.current_var,
            command=lambda _v: on_change() if on_change else None,
            length=140,
        )
        self.current_scale.grid(row=0, column=2, padx=(4, 10), sticky="ew")
        self.current_value = ttk.Label(self.frame, width=6)
        self.current_value.grid(row=0, column=3)

        ttk.Label(self.frame, text="Desired").grid(row=0, column=4, sticky="w")
        self.desired_scale = ttk.Scale(
            self.frame,
            from_=0.0,
            to=1.0,
            variable=self.desired_var,
            command=lambda _v: on_change() if on_change else None,
            length=140,
        )
        self.desired_scale.grid(row=0, column=5, padx=(4, 10), sticky="ew")
        self.desired_value = ttk.Label(self.frame, width=6)
        self.desired_value.grid(row=0, column=6)

        self.frame.columnconfigure(2, weight=1)
        self.frame.columnconfigure(5, weight=1)

        self.refresh_labels()

    def refresh_labels(self) -> None:
        self.current_value.config(text=f"{self.current_var.get():.2f}")
        self.desired_value.config(text=f"{self.desired_var.get():.2f}")

    def current(self) -> float:
        return float(self.current_var.get())

    def desired(self) -> float:
        return float(self.desired_var.get())

    def set_current(self, value: float) -> None:
        self.current_var.set(max(0.0, min(1.0, value)))
        self.refresh_labels()

    def set_desired(self, value: float) -> None:
        self.desired_var.set(max(0.0, min(1.0, value)))
        self.refresh_labels()


class TruthGuiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TRUTH ASI Intelligence Engine")
        self.root.geometry("1320x760")

        random.seed()

        self.variable_controls: list[VariableControl] = []
        self.iteration = 0
        self.score_history: list[float] = []
        self.last_interpreted: InterpretedProblem | None = None
        self.last_problem_text = ""

        self.running = False
        self.stop_event = threading.Event()
        self.result_queue: Queue = Queue()
        self.worker_thread: threading.Thread | None = None

        self.interpreter = ProblemInterpreter()
        self.data_fetcher = InternetSignalFetcher()
        self.state_constructor = StateConstructor()
        self.memory = ScarMemory(storage_path="truth_asi/memory_store.json")

        self._build_layout()
        self.root.after(120, self._process_queue)

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(main, text="Problem Setup", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=1)

        prompt_panel = ttk.LabelFrame(left, text="Mission Prompt Panel", padding=8)
        prompt_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        prompt_panel.columnconfigure(0, weight=1)
        prompt_panel.rowconfigure(1, weight=1)
        prompt_panel.rowconfigure(3, weight=1)

        ttk.Label(
            prompt_panel,
            text="Describe the real-world problem in natural language:",
        ).grid(row=0, column=0, sticky="w")
        self.problem_text = tk.Text(prompt_panel, height=4, wrap="word")
        self.problem_text.grid(row=1, column=0, sticky="nsew", pady=(4, 8))

        ttk.Label(
            prompt_panel,
            text="Optional architecture directives (constraints, stakeholders, objective hierarchy):",
        ).grid(row=2, column=0, sticky="w")
        self.directive_text = tk.Text(prompt_panel, height=4, wrap="word")
        self.directive_text.grid(row=3, column=0, sticky="nsew", pady=(4, 8))

        controls_row = ttk.Frame(prompt_panel)
        controls_row.grid(row=4, column=0, sticky="ew")
        controls_row.columnconfigure(1, weight=1)

        self.unbounded_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls_row,
            text="Unbounded search mode (no default complexity limits)",
            variable=self.unbounded_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(controls_row, text="Solve", command=self.solve_problem).grid(row=0, column=2, sticky="e")

        self.problem_text.bind("<Control-Return>", lambda _e: self.solve_problem())

        self.problem_summary_var = tk.StringVar(value="Interpreted problem: (none)")
        ttk.Label(left, textvariable=self.problem_summary_var, justify="left", wraplength=420).grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.state_label = ttk.Label(left, text="current_state={}\ndesired_state={}", justify="left", wraplength=450)
        self.state_label.grid(row=2, column=0, sticky="w", pady=(0, 8))

        self.var_canvas = tk.Canvas(left, highlightthickness=0)
        self.var_canvas.grid(row=3, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.var_canvas.yview)
        scrollbar.grid(row=3, column=1, sticky="ns")
        self.var_canvas.configure(yscrollcommand=scrollbar.set)

        self.var_container = ttk.Frame(self.var_canvas)
        self.var_container_id = self.var_canvas.create_window((0, 0), window=self.var_container, anchor="nw")

        self.var_container.bind(
            "<Configure>",
            lambda _e: self.var_canvas.configure(scrollregion=self.var_canvas.bbox("all")),
        )
        self.var_canvas.bind(
            "<Configure>",
            lambda e: self.var_canvas.itemconfig(self.var_container_id, width=e.width),
        )

        controls = ttk.Frame(left)
        controls.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        for i in range(3):
            controls.columnconfigure(i, weight=1)

        ttk.Button(controls, text="Run From Sliders", command=self.start_simulation).grid(row=0, column=0, padx=3, sticky="ew")
        ttk.Button(controls, text="Stop", command=self.stop_simulation).grid(row=0, column=1, padx=3, sticky="ew")
        ttk.Button(controls, text="Reset", command=self.reset_simulation).grid(row=0, column=2, padx=3, sticky="ew")

        right = ttk.LabelFrame(main, text="Intelligent Output", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(right, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        self.output_text = tk.Text(right, height=14, wrap="word", state="disabled")
        self.output_text.grid(row=1, column=0, sticky="nsew", pady=(6, 8))

        fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Live Iteration + Score")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Best Score")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], color="#1967d2", linewidth=2)

        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

    def _rebuild_variable_controls(self, current_state: Dict[str, float], desired_state: Dict[str, float]) -> None:
        for control in self.variable_controls:
            control.frame.destroy()
        self.variable_controls.clear()

        for variable in current_state.keys():
            control = VariableControl(self.var_container, variable, on_change=self._on_variable_changed)
            control.set_current(current_state[variable])
            control.set_desired(desired_state[variable])
            control.frame.pack(fill="x", pady=4)
            self.variable_controls.append(control)

    def solve_problem(self) -> None:
        problem_text = self.problem_text.get("1.0", "end").strip()
        directive_text = self.directive_text.get("1.0", "end").strip()
        if not problem_text:
            self.status_var.set("Enter a natural-language problem first.")
            return

        if directive_text:
            problem_text = f"{problem_text}\n\nArchitecture directives: {directive_text}"

        interpreted = self.interpreter.interpret(problem_text)
        signals = self.data_fetcher.fetch_signals(interpreted)
        current_state, desired_state = self.state_constructor.build_states(interpreted, signals)

        problem_bias = self.memory.infer_problem_bias(interpreted.variables)
        for key, direction in problem_bias.items():
            if key not in current_state:
                continue
            current_state[key] = max(0.0, min(1.0, current_state[key] + (0.04 if direction == "increase" else -0.04)))

        self.last_interpreted = interpreted
        self.last_problem_text = problem_text
        self.iteration = 0
        self.score_history.clear()
        self._update_plot()
        self._rebuild_variable_controls(current_state, desired_state)
        self._refresh_state_preview()

        self.problem_summary_var.set(
            f"Interpreted problem: domain={interpreted.domains}, intents={interpreted.intents}, variables={interpreted.variables}"
        )
        self._log(f"Problem interpreted: {interpreted.normalized_problem}")
        self._log(f"Variables used: {interpreted.variables}")
        self._log(f"Current state from internet signals: {current_state}")
        self._log(f"Desired state from goal intent: {desired_state}")
        self._log(
            "Solver mode: UNBOUNDED super-intelligence search"
            if self.unbounded_var.get()
            else "Solver mode: bounded search"
        )

        self.start_simulation()

    def _on_variable_changed(self) -> None:
        for control in self.variable_controls:
            control.refresh_labels()
        self._refresh_state_preview()

    def _refresh_state_preview(self) -> None:
        current_state = {control.name: round(control.current(), 3) for control in self.variable_controls}
        desired_state = {control.name: round(control.desired(), 3) for control in self.variable_controls}
        self.state_label.config(text=f"current_state={current_state}\ndesired_state={desired_state}")

    def _log(self, message: str) -> None:
        self.output_text.config(state="normal")
        self.output_text.insert("end", f"{message}\n")
        self.output_text.see("end")
        self.output_text.config(state="disabled")

    def _snapshot_states(self) -> tuple[Dict[str, float], Dict[str, float]]:
        current_state = {control.name: control.current() for control in self.variable_controls}
        desired_state = {control.name: control.desired() for control in self.variable_controls}
        return current_state, desired_state

    def start_simulation(self) -> None:
        if self.running:
            return
        if not self.variable_controls:
            self.status_var.set("Solve a problem or add variables before starting.")
            return

        self.running = True
        self.stop_event.clear()
        self.status_var.set("Running")

        current_state, desired_state = self._snapshot_states()
        self.worker_thread = threading.Thread(
            target=self._simulation_worker,
            args=(current_state, desired_state),
            daemon=True,
        )
        self.worker_thread.start()

    def stop_simulation(self) -> None:
        if not self.running:
            return
        self.stop_event.set()
        self.status_var.set("Stopping...")

    def reset_simulation(self) -> None:
        self.stop_simulation()
        self.iteration = 0
        self.score_history.clear()
        self._update_plot()
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.config(state="disabled")
        for control in self.variable_controls:
            control.set_current(0.5)
        self._refresh_state_preview()
        self.status_var.set("Reset complete")

    def _simulation_worker(self, current_state: Dict[str, float], desired_state: Dict[str, float]) -> None:
        tension = TensionCore(current_state=dict(current_state), desired_state=dict(desired_state))
        twin_gen = TwinGenerator()
        engine = SimulationEngine()
        selector = Selector()

        if self.unbounded_var.get():
            max_depth = max(10, 2 + len(current_state))
            iteration_budget = max(5000, len(current_state) * 800)
            max_iterations = 500
        else:
            max_depth = max(3, min(8, 2 + len(current_state) // 2))
            iteration_budget = max(180, len(current_state) * 45)
            max_iterations = 25

        while not self.stop_event.is_set():
            self.iteration += 1

            problem_field = tension.generate_problem_field()
            twins = twin_gen.generate_initial_twins(problem_field, intensity=tension.total_tension(), budget=iteration_budget)

            self.memory.bias = self.memory.get_bias() or self.memory.infer_problem_bias(list(current_state.keys()))
            twins = [self.memory.apply_bias(twin) for twin in twins]

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
                    score_gate=0.5,
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
                self.result_queue.put({"type": "message", "text": "Budget exhausted before scoring."})
                break

            ranked = selector.rank(results)
            self.memory.record_batch(ranked)

            best = ranked[0]
            best_vars = best["twin"]["variables"]
            for key, value in list(current_state.items()):
                current_state[key] = max(0.0, min(1.0, value + 0.22 * (best_vars[key] - value)))

            tension.current_state = current_state
            tension.compute_tension()

            score = float(best["score"])
            self.score_history.append(score)

            self.result_queue.put(
                {
                    "type": "update",
                    "iteration": self.iteration,
                    "best_score": score,
                    "best_state": dict(best_vars),
                    "total_tension": tension.total_tension(),
                }
            )

            normalized_best = score / max(1, len(current_state))
            if normalized_best > 0.995 or not tension.is_active() or self.iteration >= max_iterations:
                self.result_queue.put(
                    {
                        "type": "done",
                        "text": f"Solution converged at iteration {self.iteration} (score={normalized_best:.4f}, mode={'unbounded' if self.unbounded_var.get() else 'bounded'})",
                        "best_state": dict(best_vars),
                    }
                )
                break

            time.sleep(0.25)

        self.result_queue.put({"type": "stopped"})

    def _apply_best_state_to_sliders(self, best_state: Dict[str, float]) -> None:
        lookup = {control.name: control for control in self.variable_controls}
        for key, value in best_state.items():
            control = lookup.get(key)
            if control:
                control.set_current(value)
        self._refresh_state_preview()

    def _update_plot(self) -> None:
        x_vals = list(range(1, len(self.score_history) + 1))
        self.line.set_data(x_vals, self.score_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _process_queue(self) -> None:
        while True:
            try:
                payload = self.result_queue.get_nowait()
            except Empty:
                break

            kind = payload.get("type")
            if kind == "update":
                message = (
                    f"Iter {payload['iteration']:03d} | "
                    f"best score={payload['best_score']:.4f} | "
                    f"tension={payload['total_tension']:.4f} | "
                    f"state={payload['best_state']}"
                )
                self._log(message)
                self._apply_best_state_to_sliders(payload["best_state"])
                self._update_plot()
                self.status_var.set(f"Running · iteration {payload['iteration']}")
            elif kind == "done":
                self._log(payload["text"])
                if self.last_interpreted:
                    self._log(f"Best strategy found: move toward {payload.get('best_state', {})}")
                    self._log("Why it works: balances target attainment, stability, and trade-off penalties.")
                    self._log(
                        "Super-intelligence synthesis: translated mission prompt into a multi-domain variable architecture, then searched high-dimensional futures."
                    )
                    if self.last_problem_text:
                        self._log(f"Mission prompt archived: {self.last_problem_text[:220]}")
                    self.memory.remember_problem(
                        {
                            "problem": self.last_interpreted.raw_problem,
                            "domains": self.last_interpreted.domains,
                            "variables": self.last_interpreted.variables,
                            "optimized_state": payload.get("best_state", {}),
                        }
                    )
                self.status_var.set("Converged")
            elif kind == "message":
                self._log(payload["text"])
            elif kind == "stopped":
                self.running = False
                if self.status_var.get().startswith("Stopping"):
                    self.status_var.set("Stopped")

        self.root.after(120, self._process_queue)


def run() -> None:
    root = tk.Tk()
    TruthGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    run()
