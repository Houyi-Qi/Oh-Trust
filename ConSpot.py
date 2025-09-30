# ConSpot.py
"""
ConSpot Baseline for Spot-Only Trading in the Oh-Trust Framework
----------------------------------------------------------------
This module implements the ConSpot baseline, which allocates tasks in the
spot stage only. It scans candidate uniform payment prices and, for each
price, admits MUs (mobile users) whose valuation is at least the payment.
Each admitted MU requests its maximum number of tasks. A global resource
cap (Num_RB) is then enforced by keeping the top per-task utilities.

Paper context:
"Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with
 Smart Reputation Update over Dynamic Edge Networks" (IEEE TETC, under review 2025).

Main steps:
1) generate_candidates(): build (payment, desired-allocation) pairs by descending
   payment scan in [payment_min, payment_max];
2) select_best_candidate(): under the global resource cap, compute ESP/MU utilities
   and pick the candidate maximizing ESP utility;
3) main(): end-to-end baseline wrapper that returns a fixed-schema dict for logging.

Returned schema (fixed across the repo for easy evaluation):
{
    "method": "ConSpot",
    "ESP_Utility": float(ESP_total),
    "MU_Utility": float(MU_total),
    "FinishTaskNum": int(Fin_total),
    "NI": int(NI_total)
}
"""

from typing import Sequence, Tuple, List, Dict
import numpy as np


def generate_candidates(
    value: Sequence[float],
    payment_min: float,
    payment_max: float,
    NumTask: Sequence[int],
) -> List[Tuple[float, np.ndarray]]:
    """
    Generate candidate (payment, allocation) pairs for ConSpot.

    For each payment p in [payment_min, payment_max] (descending scan),
    include all MUs i with v_i >= p, and assign them their maximum task
    demand n_i. This forms a candidate allocation BEFORE applying the
    global resource constraint.

    Args:
        value (Sequence[float]): Unit valuation v_i of each MU (len = Num_MU).
        payment_min (float): Minimum payment to scan (inclusive).
        payment_max (float): Maximum payment to scan (inclusive).
        NumTask (Sequence[int]): Maximum tasks n_i each MU can request.

    Returns:
        List[Tuple[float, np.ndarray]]: A list of (payment, alloc), where:
            - payment: float
            - alloc: np.ndarray of shape (Num_MU,), desired tasks per MU
                     BEFORE enforcing the global resource cap.
    """
    value = np.asarray(value, dtype=float)
    NumTask = np.asarray(NumTask, dtype=int)
    Num_MU = value.shape[0]
    if NumTask.shape[0] != Num_MU:
        raise ValueError("value and NumTask must have the same length")

    # Normalize scan bounds
    lo, hi = float(payment_min), float(payment_max)
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        hi = lo + 1e-6  # avoid zero-length range

    candidates: List[Tuple[float, np.ndarray]] = []
    payment = float(hi)

    # Descending scan over payment values (step = 1.0 to match original code)
    while payment >= lo:
        alloc = np.zeros(Num_MU, dtype=int)
        # Select all MUs with v_i >= payment, assign their full demand n_i
        for i in range(Num_MU):
            if value[i] >= payment:
                alloc[i] = NumTask[i]
        candidates.append((payment, alloc))
        payment -= 1.0  # step of 1.0

    return candidates


def select_best_candidate(
    candidates: List[Tuple[float, np.ndarray]],
    value: Sequence[float],
    cost: float,
    Num_RB: int,
) -> Tuple[float, float, int, int]:
    """
    Select the best candidate by maximizing ESP utility, enforcing the global
    resource cap Num_RB. If total requested tasks exceed Num_RB, keep the
    top per-task utilities (v_i - payment).

    Args:
        candidates (List[Tuple[float, np.ndarray]]): Output of generate_candidates().
        value (Sequence[float]): Unit valuation v_i of each MU (len = Num_MU).
        cost (float): Unit cost of the ESP.
        Num_RB (int): Global resource cap (number of tasks that can be served).

    Returns:
        Tuple[float, float, int, int]:
            ESP_Utility (float): Best ESP utility across candidates.
            MU_Utility (float): MU total utility under the selected candidate.
            NI (int): Interaction count (≈ (#candidates) * (Num_MU + 1)).
            FinishTaskNum (int): Number of tasks actually served.
    """
    value = np.asarray(value, dtype=float)
    Num_MU = value.shape[0]

    ESP_best = 0.0
    MU_best = 0.0
    Finish_best = 0

    # Interaction number scales with candidate count and MUs (consistent with prior style)
    NI = len(candidates) * (Num_MU + 1)

    for payment, alloc in candidates:
        if payment <= cost:
            # Non-positive unit margin for ESP; skip this payment
            continue

        total_demand = int(np.sum(alloc))
        if total_demand <= Num_RB:
            # All requested tasks can be served
            esp = total_demand * (payment - cost)
            if esp > ESP_best:
                ESP_best = esp
                MU_best = float(np.sum(alloc * (value - payment)))
                Finish_best = total_demand
        else:
            # Demand exceeds capacity: keep top (v_i - payment) per-task utilities
            per_task_utils: List[float] = []
            for i in range(Num_MU):
                if alloc[i] > 0:
                    per_task_utils.extend([float(value[i] - payment)] * int(alloc[i]))

            if not per_task_utils:
                continue

            per_task_utils.sort(reverse=True)
            served = max(0, int(Num_RB))
            served_utils = per_task_utils[:served]

            esp = served * (payment - cost)
            mu = float(np.sum(served_utils))

            if esp > ESP_best:
                ESP_best = esp
                MU_best = mu
                Finish_best = served

    return float(ESP_best), float(MU_best), int(NI), int(Finish_best)


def main(
    value: Sequence[float],
    payment_min: float,
    payment_max: float,
    cost: float,
    Num_RB: int,
    NumTask: Sequence[int],
) -> Dict[str, float]:
    """
    End-to-end ConSpot baseline:
    1) Generate candidate (payment, allocation) pairs;
    2) Select the best one under the global resource cap.

    Args:
        value (Sequence[float]): Unit valuation per MU.
        payment_min (float): Minimum payment to scan (inclusive).
        payment_max (float): Maximum payment to scan (inclusive).
        cost (float): Unit cost of the ESP.
        Num_RB (int): Global resource cap.
        NumTask (Sequence[int]): Max tasks per MU.

    Returns:
        Dict[str, float]: Fixed-schema dictionary for logging:
            {
                "method": "ohtrust",
                "ESP_Utility": float,
                "MU_Utility": float,
                "FinishTaskNum": int,
                "NI": int
            }
    """
    candidates = generate_candidates(value, payment_min, payment_max, NumTask)
    ESP_total, MU_total, NI_total, Fin_total = select_best_candidate(
        candidates, value, cost, Num_RB
    )

    return {
        "method": "ConSpot",
        "ESP_Utility": float(ESP_total),
        "MU_Utility": float(MU_total),
        "FinishTaskNum": int(Fin_total),
        "NI": int(NI_total),
    }


if __name__ == "__main__":
    # Minimal example for quick sanity check
    v = [8.0, 10.0, 12.0, 7.5]
    cost = 6.0
    rb = 10
    maxtask = [3, 5, 4, 2]
    payment_min, payment_max = min(v), max(v)
    res = main(v, payment_min, payment_max, cost, rb, maxtask)
    print(res)
