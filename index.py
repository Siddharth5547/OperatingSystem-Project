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
