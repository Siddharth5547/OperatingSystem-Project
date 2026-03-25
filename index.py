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
