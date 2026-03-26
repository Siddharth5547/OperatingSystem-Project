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