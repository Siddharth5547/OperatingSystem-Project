"""
Energy-Efficient CPU Scheduling Simulation
=============================================
DVFS · Thermal Awareness · Workload Prediction · Heterogeneous big.LITTLE Cores
Schedulers: Custom (energy-efficient), FCFS, Round Robin
"""

from __future__ import annotations
import copy, random, math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─────────────────────────── Constants ────────────────────────────

FREQ_LOW   = 1.0   # GHz
FREQ_MED   = 2.0
FREQ_HIGH  = 3.0

TEMP_AMBIENT    = 40.0   # °C
TEMP_THRESHOLD  = 85.0   # trigger throttle / migration
COOLING_FACTOR  = 0.15   # per tick when idle
HEATING_COEFF   = 2.5    # multiplier for freq × load → ΔT

RR_QUANTUM = 2

# ─────────────────────────── Enums ────────────────────────────────

class TaskType(Enum):
    CPU = "CPU"
    IO  = "IO"

class CoreType(Enum):
    BIG    = "big"
    LITTLE = "little"

# ─────────────────────────── Task ─────────────────────────────────

@dataclass
class Task:
    id: int
    arrival_time: int
    burst_time: int
    remaining_time: int = 0
    task_type: TaskType = TaskType.CPU
    priority: int = 1          # 1 = highest priority

    def __post_init__(self):
        if self.remaining_time == 0:
            self.remaining_time = self.burst_time

    def __repr__(self):
        return f"T{self.id}"

        
# ─────────────────────────── Core ─────────────────────────────────

@dataclass
class Core:
    id: int
    core_type: CoreType
    frequency: float = FREQ_HIGH
    temperature: float = TEMP_AMBIENT
    energy: float = 0.0
    current_task: Optional[Task] = None

    # ── DVFS ──
    def set_frequency(self, level: str):
        mapping = {"LOW": FREQ_LOW, "MEDIUM": FREQ_MED, "HIGH": FREQ_HIGH}
        self.frequency = mapping.get(level, FREQ_HIGH)

    # ── Thermal model ──
    def update_temperature(self):
        if self.current_task is not None:
            load = 1.0
            self.temperature += HEATING_COEFF * self.frequency * load * 0.1
        else:
            self.temperature -= COOLING_FACTOR * (self.temperature - TEMP_AMBIENT)
        self.temperature = max(self.temperature, TEMP_AMBIENT)

    def is_overheated(self) -> bool:
        return self.temperature >= TEMP_THRESHOLD

    # ── Energy ──
    def compute_energy(self, load: float, dt: float = 1.0) -> float:
        """energy = frequency² × load × dt"""
        e = (self.frequency ** 2) * load * dt
        self.energy += e
        return e

    def label(self) -> str:
        return f"Core-{self.id} ({self.core_type.value})"

    def __repr__(self):
        return self.label()

# ─────────────────────── Workload Predictor ───────────────────────

class WorkloadPredictor:
    """Simple moving-average predictor over recent burst times."""

    def __init__(self, window: int = 5):
        self.window = window
        self.history: List[float] = []

    def record(self, burst: float):
        self.history.append(burst)

    def predict(self) -> float:
        if not self.history:
            return 3.0                 # default guess
        recent = self.history[-self.window:]
        return sum(recent) / len(recent)

# ──────────────────── Scheduling Algorithms ───────────────────────

class SchedulerBase:
    """Common bookkeeping shared by all schedulers."""

    name: str = "Base"

    def __init__(self, cores: List[Core]):
        self.cores = cores
        self.ready_queue: List[Task] = []
        self.completed: List[Task] = []
        self.logs: List[str] = []
        self.gantt: Dict[int, List[Tuple[int, int, int]]] = {c.id: [] for c in cores}
        self.temp_history: Dict[int, List[float]] = {c.id: [] for c in cores}
        self.predictor = WorkloadPredictor()

    # ── helpers ──
    def _idle_core(self, preferred_type: Optional[CoreType] = None) -> Optional[Core]:
        for c in self.cores:
            if c.current_task is None:
                if preferred_type is None or c.core_type == preferred_type:
                    return c
        # fallback: any idle core
        for c in self.cores:
            if c.current_task is None:
                return c
        return None

    def _assign(self, core: Core, task: Task, tick: int):
        core.current_task = task
        self.logs.append(f"  [t={tick}] {task} → {core.label()} @ {core.frequency} GHz")

    def _tick_cores(self, tick: int):
        """Execute one tick on every busy core; update energy & temp."""
        for c in self.cores:
            if c.current_task is not None:
                c.current_task.remaining_time -= 1
                c.compute_energy(load=1.0)
                # gantt entry
                self.gantt[c.id].append((tick, tick + 1, c.current_task.id))
                if c.current_task.remaining_time <= 0:
                    self.logs.append(f"  [t={tick}] {c.current_task} completed on {c.label()}")
                    self.completed.append(c.current_task)
                    c.current_task = None
            c.update_temperature()
            self.temp_history[c.id].append(c.temperature)

    def total_energy(self) -> float:
        return sum(c.energy for c in self.cores)

# ────────────── Custom Energy-Efficient Scheduler ─────────────────

class CustomScheduler(SchedulerBase):
    name = "Custom (Energy-Efficient)"

    def _adjust_dvfs(self):
        predicted = self.predictor.predict()
        for c in self.cores:
            if c.is_overheated():
                c.set_frequency("LOW")
            elif predicted > 5:
                c.set_frequency("HIGH" if c.core_type == CoreType.BIG else "MEDIUM")
            elif predicted > 3:
                c.set_frequency("MEDIUM")
            else:
                c.set_frequency("LOW")

    def _handle_thermal(self, tick: int):
        """Migrate tasks from overheated cores when possible."""
        for c in self.cores:
            if c.is_overheated() and c.current_task is not None:
                target = None
                for other in self.cores:
                    if other.id != c.id and other.current_task is None and not other.is_overheated():
                        target = other
                        break
                if target:
                    task = c.current_task
                    c.current_task = None
                    target.current_task = task
                    self.logs.append(
                        f"  [t={tick}] THERMAL MIGRATION: {task} from {c.label()} → {target.label()}"
                    )

    def schedule(self, tick: int):
        self._adjust_dvfs()
        self._handle_thermal(tick)
        # Sort: higher priority first (lower number), then shorter remaining
        self.ready_queue.sort(key=lambda t: (t.priority, t.remaining_time))

        still_waiting: List[Task] = []
        for task in self.ready_queue:
            self.predictor.record(task.burst_time)
            if task.task_type == TaskType.CPU:
                core = self._idle_core(CoreType.BIG)
            else:
                core = self._idle_core(CoreType.LITTLE)
            if core:
                self._assign(core, task, tick)
            else:
                still_waiting.append(task)
        self.ready_queue = still_waiting

    def run(self, tasks: List[Task], max_ticks: int = 100) -> None:
        tasks = sorted(tasks, key=lambda t: t.arrival_time)
        task_idx = 0
        for tick in range(max_ticks):
            # 1. Inject arrivals
            while task_idx < len(tasks) and tasks[task_idx].arrival_time <= tick:
                self.ready_queue.append(tasks[task_idx])
                task_idx += 1
            # 2-4. Schedule
            self.schedule(tick)
            # 5-6. Execute
            self._tick_cores(tick)
            # early exit
            if not self.ready_queue and all(c.current_task is None for c in self.cores) and task_idx >= len(tasks):
                # pad temp history so all cores have same length
                for c in self.cores:
                    pass
                break

# ─────────────────────────── FCFS ─────────────────────────────────

class FCFSScheduler(SchedulerBase):
    name = "FCFS"

    def run(self, tasks: List[Task], max_ticks: int = 100) -> None:
        tasks = sorted(tasks, key=lambda t: t.arrival_time)
        task_idx = 0
        for tick in range(max_ticks):
            while task_idx < len(tasks) and tasks[task_idx].arrival_time <= tick:
                self.ready_queue.append(tasks[task_idx])
                task_idx += 1

            for task in list(self.ready_queue):
                core = self._idle_core()
                if core:
                    self._assign(core, task, tick)
                    self.ready_queue.remove(task)

            self._tick_cores(tick)

            if not self.ready_queue and all(c.current_task is None for c in self.cores) and task_idx >= len(tasks):
                break

# ────────────────────────── Round Robin ───────────────────────────

class RRScheduler(SchedulerBase):
    name = "Round Robin"

    def __init__(self, cores: List[Core], quantum: int = RR_QUANTUM):
        super().__init__(cores)
        self.quantum = quantum
        self.time_on_core: Dict[int, int] = {c.id: 0 for c in cores}

    def run(self, tasks: List[Task], max_ticks: int = 100) -> None:
        tasks = sorted(tasks, key=lambda t: t.arrival_time)
        task_idx = 0
        for tick in range(max_ticks):
            # arrivals
            while task_idx < len(tasks) and tasks[task_idx].arrival_time <= tick:
                self.ready_queue.append(tasks[task_idx])
                task_idx += 1

            # preempt on quantum expiry
            for c in self.cores:
                if c.current_task is not None:
                    self.time_on_core[c.id] += 1
                    if self.time_on_core[c.id] >= self.quantum and c.current_task.remaining_time > 0:
                        self.logs.append(
                            f"  [t={tick}] PREEMPT {c.current_task} on {c.label()} (quantum expired)"
                        )
                        self.ready_queue.append(c.current_task)
                        c.current_task = None
                        self.time_on_core[c.id] = 0

            # assign from queue
            for task in list(self.ready_queue):
                core = self._idle_core()
                if core:
                    self._assign(core, task, tick)
                    self.ready_queue.remove(task)
                    self.time_on_core[core.id] = 0

            self._tick_cores(tick)

            if not self.ready_queue and all(c.current_task is None for c in self.cores) and task_idx >= len(tasks):
                break

# ──────────────────── Visualization Helpers ───────────────────────

TASK_COLORS = plt.cm.tab20.colors  # 20 distinct colours

def draw_gantt(scheduler: SchedulerBase, ax: plt.Axes):
    ax.set_title(f"Gantt Chart – {scheduler.name}", fontweight="bold")
    yticks, ylabels = [], []
    for idx, core in enumerate(scheduler.cores):
        y = idx
        yticks.append(y)
        ylabels.append(core.label())
        for start, end, tid in scheduler.gantt[core.id]:
            color = TASK_COLORS[tid % len(TASK_COLORS)]
            ax.barh(y, end - start, left=start, height=0.6, color=color,
                    edgecolor="black", linewidth=0.5)
            ax.text((start + end) / 2, y, f"T{tid}", ha="center", va="center", fontsize=7)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (ticks)")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)


def draw_energy_comparison(schedulers: List[SchedulerBase], ax: plt.Axes):
    names = [s.name for s in schedulers]
    energies = [s.total_energy() for s in schedulers]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax.bar(names, energies, color=colors[:len(names)], edgecolor="black", linewidth=0.5)
    for bar, e in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{e:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total Energy (freq² × load × t)")
    ax.set_title("Energy Comparison", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def draw_temperature(scheduler: SchedulerBase, ax: plt.Axes):
    ax.set_title(f"Core Temperature – {scheduler.name}", fontweight="bold")
    for core in scheduler.cores:
        temps = scheduler.temp_history[core.id]
        ax.plot(range(len(temps)), temps, label=core.label())
    ax.axhline(TEMP_THRESHOLD, color="red", linestyle="--", linewidth=0.8, label="Threshold")
    ax.set_xlabel("Time (ticks)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)

    # ─────────────────────── Sample Tasks ─────────────────────────────

def make_sample_tasks() -> List[Task]:
    return [
        Task(id=0,  arrival_time=0,  burst_time=6,  task_type=TaskType.CPU, priority=2),
        Task(id=1,  arrival_time=0,  burst_time=3,  task_type=TaskType.IO,  priority=1),
        Task(id=2,  arrival_time=1,  burst_time=8,  task_type=TaskType.CPU, priority=3),
        Task(id=3,  arrival_time=1,  burst_time=2,  task_type=TaskType.IO,  priority=1),
        Task(id=4,  arrival_time=2,  burst_time=5,  task_type=TaskType.CPU, priority=2),
        Task(id=5,  arrival_time=3,  burst_time=4,  task_type=TaskType.IO,  priority=3),
        Task(id=6,  arrival_time=4,  burst_time=7,  task_type=TaskType.CPU, priority=1),
        Task(id=7,  arrival_time=5,  burst_time=3,  task_type=TaskType.CPU, priority=2),
        Task(id=8,  arrival_time=6,  burst_time=2,  task_type=TaskType.IO,  priority=1),
        Task(id=9,  arrival_time=7,  burst_time=9,  task_type=TaskType.CPU, priority=3),
    ]


def make_cores() -> List[Core]:
    return [
        Core(id=0, core_type=CoreType.BIG),
        Core(id=1, core_type=CoreType.BIG),
        Core(id=2, core_type=CoreType.LITTLE, frequency=FREQ_MED),
        Core(id=3, core_type=CoreType.LITTLE, frequency=FREQ_LOW),
    ]

# ───────────────────────── main() ─────────────────────────────────

def main():
    print("=" * 65)
    print("  Energy-Efficient CPU Scheduling Simulation")
    print("  DVFS · Thermal Awareness · Workload Prediction")
    print("=" * 65)

    # ── Run each scheduler on an independent copy of tasks/cores ──
    schedulers: List[SchedulerBase] = []

    for SchedulerCls in (CustomScheduler, FCFSScheduler, RRScheduler):
        tasks = make_sample_tasks()
        cores = make_cores()
        if SchedulerCls == RRScheduler:
            sched = SchedulerCls(cores, quantum=RR_QUANTUM)
        else:
            sched = SchedulerCls(cores)

        print(f"\n{'─' * 60}")
        print(f"  Scheduler: {sched.name}")
        print(f"{'─' * 60}")

        sched.run(tasks, max_ticks=100)

        for line in sched.logs:
            print(line)

        print(f"\n  Total energy consumed: {sched.total_energy():.2f}")
        print(f"  Tasks completed:      {len(sched.completed)}")
        schedulers.append(sched)

    # ── Visualisation ──
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("CPU Scheduling Simulation Results", fontsize=15, fontweight="bold")

    # Row 1: Gantt charts (one per scheduler)
    for i, sched in enumerate(schedulers):
        ax = fig.add_subplot(3, 3, i + 1)
        draw_gantt(sched, ax)

    # Row 2 col 1-2: Temperature for Custom scheduler
    ax_temp = fig.add_subplot(3, 3, 4)
    draw_temperature(schedulers[0], ax_temp)

    # Row 2 col 2: Temperature for FCFS
    ax_temp2 = fig.add_subplot(3, 3, 5)
    draw_temperature(schedulers[1], ax_temp2)

    # Row 2 col 3: Temperature for RR
    ax_temp3 = fig.add_subplot(3, 3, 6)
    draw_temperature(schedulers[2], ax_temp3)

    # Row 3: Energy comparison (span full width)
    ax_energy = fig.add_subplot(3, 1, 3)
    draw_energy_comparison(schedulers, ax_energy)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = "scheduling_results.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Results saved to {output_path}")
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    main()
