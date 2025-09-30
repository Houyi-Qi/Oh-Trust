# BiN_TCD.py
# ----------------------------------------------------
# BiN_TCD: Spot trading (ConSpot) for Oh-Trust
# - Scans candidate payments from payment_max down to payment_min
# - Under a global resource constraint (Num_RB), selects the price and
#   allocation that maximize ESP's utility
# ----------------------------------------------------
"""
BiN_TCD: Spot Trading Mechanism (ConSpot)
-----------------------------------------
Implements the ConSpot baseline for Oh-Trust:
  • Iterates payment price from high → low
  • For each price, computes desired allocations based on user valuations
  • Checks feasibility under ESP’s remaining resource capacity (Num_RB)
  • Computes ESP and MU utilities, selecting the best outcome

Returns a tuple:
  (ESP_Utility, MU_Utility, NI, FinishTaskNum)
"""

from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np


def conspot(
    value: Sequence[float],
    NumTask: Sequence[int],
    cost: float,
    payment_min: float,
    payment_max: float,
    Num_RB: int,
) -> Tuple[float, float, int, int]:
    """
    End-to-end ConSpot spot trading.

    Args:
        value (Sequence[float]): Per-task valuation of each MU (length = Num_MU).
        NumTask (Sequence[int]): Maximum number of tasks per MU (length = Num_MU).
        cost (float): ESP's per-task cost.
        payment_min (float): Minimum payment price to consider.
        payment_max (float): Maximum payment price to consider.
        Num_RB (int): ESP's available resource blocks (capacity).

    Returns:
        Tuple[float, float, int, int]:
            ESP_Utility   (float): ESP's maximum achieved utility.
            MU_Utility    (float): MUs' total achieved utility.
            NI            (int): Number of interactions (for complexity accounting).
            FinishTaskNum (int): Total number of tasks completed under optimal price.
    """
    v = np.asarray(value, dtype=float)
    n = np.asarray(NumTask, dtype=int)
    Num_MU = v.shape[0]
    NI = 0

    ESP_best = 0.0
    MU_best = 0.0
    Fin_best = 0

    # Scan price grid from high → low
    p = float(payment_max)
    while p >= payment_min:
        desired = np.where(v >= p, n, 0)  # tasks MUs want at this price
        NI += (Num_MU + 1)

        total = int(np.sum(desired))
        if p > cost and total <= Num_RB:
            # Feasible: allocate all desired tasks
            esp = total * (p - cost)
            if esp > ESP_best:
                ESP_best = esp
                MU_best = float(np.sum(desired * (v - p)))
                Fin_best = total
        elif p > cost and total > Num_RB:
            # Over-demand: allocate only top (v_i - p) tasks
            per_utils = []
            for i in range(Num_MU):
                if desired[i] > 0:
                    per_utils.extend([v[i] - p] * int(desired[i]))
            per_utils.sort(reverse=True)
            per_utils = per_utils[:Num_RB]
            esp = Num_RB * (p - cost)
            mu = float(np.sum(per_utils))
            if esp > ESP_best:
                ESP_best = esp
                MU_best = mu
                Fin_best = int(Num_RB)

        p -= 1.0  # decrease price step by 1

    return float(ESP_best), float(MU_best), int(NI), int(Fin_best)
