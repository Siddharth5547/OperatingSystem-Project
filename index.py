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