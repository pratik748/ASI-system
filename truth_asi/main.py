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
from memory.scar import ScarMemory
from simulation.engine import SimulationEngine
from simulation.selector import Selector
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


class TruthGuiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TRUTH ASI Control Panel")
        self.root.geometry("1200x680")

        random.seed()

        self.variable_controls: list[VariableControl] = []
        self.iteration = 0
        self.score_history: list[float] = []

        self.running = False
        self.stop_event = threading.Event()
        self.result_queue: Queue = Queue()
        self.worker_thread: threading.Thread | None = None

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
        left.rowconfigure(2, weight=1)

        add_row = ttk.Frame(left)
        add_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        add_row.columnconfigure(0, weight=1)

        self.variable_name_var = tk.StringVar()
        self.name_entry = ttk.Entry(add_row, textvariable=self.variable_name_var)
        self.name_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.name_entry.bind("<Return>", lambda _e: self.add_variable())

        ttk.Button(add_row, text="Add Variable", command=self.add_variable).grid(row=0, column=1)

        self.state_label = ttk.Label(left, text="current_state={}\ndesired_state={}", justify="left")
        self.state_label.grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.var_canvas = tk.Canvas(left, highlightthickness=0)
        self.var_canvas.grid(row=2, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.var_canvas.yview)
        scrollbar.grid(row=2, column=1, sticky="ns")
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
        controls.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for i in range(3):
            controls.columnconfigure(i, weight=1)

        ttk.Button(controls, text="Start Simulation", command=self.start_simulation).grid(row=0, column=0, padx=3, sticky="ew")
        ttk.Button(controls, text="Stop", command=self.stop_simulation).grid(row=0, column=1, padx=3, sticky="ew")
        ttk.Button(controls, text="Reset", command=self.reset_simulation).grid(row=0, column=2, padx=3, sticky="ew")

        right = ttk.LabelFrame(main, text="Live Output", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(right, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        self.output_text = tk.Text(right, height=12, wrap="word", state="disabled")
        self.output_text.grid(row=1, column=0, sticky="nsew", pady=(6, 8))

        fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Score vs Iteration")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Best Score")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], color="#1967d2", linewidth=2)

        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

    def add_variable(self) -> None:
        raw_name = self.variable_name_var.get().strip()
        if not raw_name:
            return

        existing = {control.name for control in self.variable_controls}
        if raw_name in existing:
            self.status_var.set(f"Variable '{raw_name}' already exists.")
            return

        control = VariableControl(self.var_container, raw_name, on_change=self._on_variable_changed)
        control.frame.pack(fill="x", pady=4)
        self.variable_controls.append(control)
        self.variable_name_var.set("")
        self._refresh_state_preview()

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
            self.status_var.set("Add at least one variable before starting.")
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
        memory = ScarMemory()

        max_depth = 3
        iteration_budget = 250

        while not self.stop_event.is_set():
            self.iteration += 1

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
                self.result_queue.put({"type": "message", "text": "Budget exhausted before scoring."})
                break

            ranked = selector.rank(results)
            memory.record_batch(ranked)
            memory.get_bias()

            best = ranked[0]
            best_vars = best["twin"]["variables"]
            for key, value in list(current_state.items()):
                current_state[key] = max(0.0, min(1.0, value + 0.15 * (best_vars[key] - value)))

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
            if normalized_best > 0.98 or not tension.is_active():
                self.result_queue.put(
                    {
                        "type": "done",
                        "text": f"Convergence reached at iteration {self.iteration} (score={normalized_best:.4f})",
                    }
                )
                break

            time.sleep(0.75)

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
    app = TruthGuiApp(root)
    app.variable_name_var.set("x")
    app.add_variable()
    root.mainloop()


if __name__ == "__main__":
    run()
