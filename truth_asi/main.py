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

try:
    from .core.tension import TensionCore
    from .data.signals import InternetSignalFetcher
    from .memory.scar import ScarMemory
    from .problem.interpreter import InterpretedProblem, ProblemInterpreter
    from .simulation.engine import SimulationEngine
    from .simulation.selector import Selector
    from .state.constructor import StateConstructor
    from .twins.generator import TwinGenerator
except ImportError:  # pragma: no cover - allows running as a script
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
        self.root.title("TRUTH ASI // MACHINE GOD CONSOLE")
        self.root.geometry("1360x820")

        random.seed()

        self.variable_controls: list[VariableControl] = []
        self.iteration = 0
        self.score_history: list[float] = []
        self.last_interpreted: InterpretedProblem | None = None
        self.thinking_by_iteration: Dict[int, str] = {}

        self.running = False
        self.stop_event = threading.Event()
        self.result_queue: Queue = Queue()
        self.worker_thread: threading.Thread | None = None

        self.interpreter = ProblemInterpreter()
        self.data_fetcher = InternetSignalFetcher()
        self.state_constructor = StateConstructor()
        self.memory = ScarMemory(storage_path="truth_asi/memory_store.json")

        self._configure_styles()
        self._build_layout()
        self.root.after(120, self._process_queue)

    def _configure_styles(self) -> None:
        self.root.configure(bg="#05070f")
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background="#05070f", foreground="#d6e9ff", fieldbackground="#0b1020")
        style.configure("TFrame", background="#05070f")
        style.configure("TLabelframe", background="#060b16", foreground="#7bf5ff", borderwidth=1)
        style.configure("TLabelframe.Label", background="#060b16", foreground="#7bf5ff", font=("TkDefaultFont", 10, "bold"))
        style.configure("TLabel", background="#05070f", foreground="#d6e9ff")
        style.configure("TEntry", fieldbackground="#0b1020", foreground="#d6e9ff", insertcolor="#7bf5ff")
        style.configure("TButton", background="#111b3a", foreground="#9cf6ff", borderwidth=1)
        style.map("TButton", background=[("active", "#1e2b57")], foreground=[("active", "#c6fcff")])
        style.configure("Vertical.TScrollbar", background="#10182e", troughcolor="#090f1d")

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(main, text="Simulation Inputs", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=1)

        input_row = ttk.Frame(left)
        input_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        input_row.columnconfigure(0, weight=1)

        ttk.Label(input_row, text="Inject Problem Into Core").grid(row=0, column=0, sticky="w")
        self.problem_var = tk.StringVar()
        self.problem_entry = ttk.Entry(input_row, textvariable=self.problem_var)
        self.problem_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(4, 0))
        self.problem_entry.bind("<Return>", lambda _e: self.solve_problem())
        ttk.Button(input_row, text="Awaken", command=self.solve_problem).grid(row=1, column=1, sticky="ew", pady=(4, 0))

        self.problem_summary_var = tk.StringVar(value="Interpreted problem: (none)")
        ttk.Label(left, textvariable=self.problem_summary_var, justify="left", wraplength=420).grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.state_label = ttk.Label(left, text="current_state={}\ndesired_state={}", justify="left", wraplength=450)
        self.state_label.grid(row=2, column=0, sticky="w", pady=(0, 8))

        self.var_canvas = tk.Canvas(left, highlightthickness=0, bg="#060b16")
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

        right = ttk.LabelFrame(main, text="Machine Mind", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=2)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(right, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

        self.output_text = tk.Text(
            right,
            height=11,
            wrap="word",
            state="disabled",
            bg="#070e1f",
            fg="#94f8ff",
            insertbackground="#94f8ff",
            selectbackground="#1c3a65",
            relief="flat",
        )
        self.output_text.grid(row=1, column=0, sticky="nsew", pady=(6, 8))

        introspect = ttk.LabelFrame(right, text="Cognition Feed (click iteration)", padding=8)
        introspect.grid(row=2, column=0, sticky="nsew")
        introspect.columnconfigure(0, weight=1)
        introspect.columnconfigure(1, weight=2)
        introspect.rowconfigure(0, weight=1)

        self.iteration_list = tk.Listbox(
            introspect,
            bg="#090f1d",
            fg="#f2fbff",
            selectbackground="#1d3563",
            selectforeground="#d7ffff",
            activestyle="none",
        )
        self.iteration_list.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.iteration_list.bind("<<ListboxSelect>>", self._on_iteration_selected)

        self.thinking_text = tk.Text(
            introspect,
            wrap="word",
            state="disabled",
            bg="#050913",
            fg="#7bf5ff",
            insertbackground="#7bf5ff",
            relief="flat",
        )
        self.thinking_text.grid(row=0, column=1, sticky="nsew")

        fig = Figure(figsize=(6, 3), dpi=100, facecolor="#05070f")
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor("#070e1f")
        self.ax.set_title("Iteration Power Curve", color="#b8f8ff")
        self.ax.set_xlabel("Iteration", color="#b8f8ff")
        self.ax.set_ylabel("Best Score", color="#b8f8ff")
        self.ax.tick_params(colors="#89d7ff")
        self.ax.grid(True, alpha=0.25, color="#2d4f74")
        self.line, = self.ax.plot([], [], color="#2ee2ff", linewidth=2)

        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        right.rowconfigure(3, weight=2)

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
        problem_text = self.problem_var.get().strip()
        if not problem_text:
            self.status_var.set("Enter a natural-language problem first.")
            return

        interpreted = self.interpreter.interpret(problem_text)
        signals = self.data_fetcher.fetch_signals(interpreted)
        current_state, desired_state = self.state_constructor.build_states(interpreted, signals)

        problem_bias = self.memory.infer_problem_bias(interpreted.variables)
        for key, direction in problem_bias.items():
            if key not in current_state:
                continue
            current_state[key] = max(0.0, min(1.0, current_state[key] + (0.04 if direction == "increase" else -0.04)))

        self.last_interpreted = interpreted
        self.iteration = 0
        self.score_history.clear()
        self.thinking_by_iteration.clear()
        self.iteration_list.delete(0, "end")
        self._set_thinking_preview("Awaiting cognition snapshots...")
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

        self.start_simulation()

    def _on_variable_changed(self) -> None:
        for control in self.variable_controls:
            control.refresh_labels()
        self._refresh_state_preview()

    def _refresh_state_preview(self) -> None:
        current_state = {control.name: round(control.current(), 3) for control in self.variable_controls}
        desired_state = {control.name: round(control.desired(), 3) for control in self.variable_controls}
        self.state_label.config(text=f"current_state={current_state}\ndesired_state={desired_state}")

    def _set_thinking_preview(self, message: str) -> None:
        self.thinking_text.config(state="normal")
        self.thinking_text.delete("1.0", "end")
        self.thinking_text.insert("end", message)
        self.thinking_text.config(state="disabled")

    def _on_iteration_selected(self, _event: tk.Event) -> None:
        selected = self.iteration_list.curselection()
        if not selected:
            return

        line = self.iteration_list.get(selected[0])
        if not line.startswith("Iter "):
            return

        try:
            iteration = int(line.split()[1])
        except (ValueError, IndexError):
            return

        thought = self.thinking_by_iteration.get(iteration, "No thinking trace captured for this step.")
        self._set_thinking_preview(thought)

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
        self.thinking_by_iteration.clear()
        self.iteration_list.delete(0, "end")
        self._set_thinking_preview("Cognition feed reset.")
        self._update_plot()
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.config(state="disabled")
        for control in self.variable_controls:
            control.set_current(0.5)
        self._refresh_state_preview()
        self.status_var.set("Reset complete")

    def _build_thinking_trace(
        self,
        iteration: int,
        best_state: Dict[str, float],
        desired_state: Dict[str, float],
        tension_total: float,
        score: float,
    ) -> str:
        deltas = sorted(
            ((key, desired_state[key] - value) for key, value in best_state.items()),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        top_focus = ", ".join(f"{k}:{d:+.3f}" for k, d in deltas[:4])
        mode = "stabilize" if tension_total < 0.35 else "push target acquisition"
        return (
            f"Iteration {iteration}\n"
            f"Neural posture: {mode}\n"
            f"Best score: {score:.4f}\n"
            f"Residual tension: {tension_total:.4f}\n"
            f"Largest target gaps: {top_focus}\n"
            "Interpretation: the engine is reweighting variables toward the target while preserving system stability."
        )

    def _simulation_worker(self, current_state: Dict[str, float], desired_state: Dict[str, float]) -> None:
        tension = TensionCore(current_state=dict(current_state), desired_state=dict(desired_state))
        twin_gen = TwinGenerator()
        engine = SimulationEngine()
        selector = Selector()

        max_depth = max(3, min(8, 2 + len(current_state) // 2))
        iteration_budget = max(180, len(current_state) * 45)

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
            thought = self._build_thinking_trace(
                iteration=self.iteration,
                best_state=dict(best_vars),
                desired_state=dict(desired_state),
                tension_total=tension.total_tension(),
                score=score,
            )

            self.result_queue.put(
                {
                    "type": "update",
                    "iteration": self.iteration,
                    "best_score": score,
                    "best_state": dict(best_vars),
                    "total_tension": tension.total_tension(),
                    "thought": thought,
                }
            )

            normalized_best = score / max(1, len(current_state))
            if normalized_best > 0.985 or not tension.is_active() or self.iteration >= 25:
                self.result_queue.put(
                    {
                        "type": "done",
                        "text": f"Solution converged at iteration {self.iteration} (score={normalized_best:.4f})",
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
                iteration = payload["iteration"]
                message = (
                    f"Iter {iteration:03d} | "
                    f"best score={payload['best_score']:.4f} | "
                    f"tension={payload['total_tension']:.4f} | "
                    f"state={payload['best_state']}"
                )
                self._log(message)
                self.thinking_by_iteration[iteration] = payload.get("thought", "No thought available.")
                self.iteration_list.insert("end", f"Iter {iteration}")
                self.iteration_list.yview_moveto(1)
                if len(self.thinking_by_iteration) == 1:
                    self.iteration_list.selection_clear(0, "end")
                    self.iteration_list.selection_set(0)
                    self._set_thinking_preview(self.thinking_by_iteration[iteration])
                self._apply_best_state_to_sliders(payload["best_state"])
                self._update_plot()
                self.status_var.set(f"Running · iteration {iteration}")
            elif kind == "done":
                self._log(payload["text"])
                if self.last_interpreted:
                    self._log(f"Best strategy found: move toward {payload.get('best_state', {})}")
                    self._log("Why it works: balances target attainment, stability, and trade-off penalties.")
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
    app = TruthGuiApp(root)
    app.problem_var.set("How do I maximize business growth with low risk?")
    root.mainloop()


if __name__ == "__main__":
    run()
