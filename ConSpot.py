# baselines/con_spot.py
import numpy as np
from typing import Sequence, Tuple, List

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
    demand n_i. This forms a candidate allocation before applying the
    global resource constraint.

    Args:
        value: array-like, unit valuation v_i of each MU (len = Num_MU)
        payment_min: minimum payment to scan
        payment_max: maximum payment to scan
        NumTask: array-like, maximum tasks n_i each MU can request

    Returns:
        List of tuples (payment, alloc), where:
          - payment: float
          - alloc: np.ndarray of shape (Num_MU,), desired tasks per MU
                   BEFORE enforcing the global resource cap.
    """
    value = np.asarray(value, dtype=float)
    NumTask = np.asarray(NumTask, dtype=int)
    Num_MU = value.shape[0]
    if NumTask.shape[0] != Num_MU:
        raise ValueError("value and NumTask must have the same length")

    candidates: List[Tuple[float, np.ndarray]] = []
    payment = float(payment_max)

    # Descending scan over payment values
    while payment >= payment_min:
        alloc = np.zeros(Num_MU, dtype=int)
        # Select all MUs with v_i >= payment, assign their full demand n_i
        for i in range(Num_MU):
            if value[i] >= payment:
                alloc[i] = NumTask[i]
        candidates.append((payment, alloc))
        payment -= 1.0  # step of 1, consistent with the original code

    return candidates


def select_best_candidate(
    candidates: List[Tuple[float, np.ndarray]],
    value: Sequence[float],
    cost: float,
    Num_RB: int,
) -> Tuple[float, float, int, int]:
    """
    Select the best candidate by maximizing ESP utility, enforcing the global
    resource cap Num_RB. If total requested tasks exceed Num_RB, pick the
    top-utility tasks (v_i - payment) at the per-task level.

    Args:
        candidates: list produced by generate_candidates()
        value: array-like, unit valuation v_i of each MU (len = Num_MU)
        cost: float, unit cost of ESP
        Num_RB: int, global resource cap (number of tasks that can be served)

    Returns:
        ESP_Utility: float, max ESP utility across candidates
        MU_Utility: float, MU total utility under the selected candidate
        NI: int, interaction count (≈ (#candidates) * (Num_MU + 1))
        FinishTaskNum: int, number of tasks actually served under the selected candidate
    """
    value = np.asarray(value, dtype=float)
    Num_MU = value.shape[0]

    ESP_best = 0.0
    MU_best = 0.0
    Finish_best = 0

    # Interaction number scales with candidate count and MUs (match prior style)
    NI = len(candidates) * (Num_MU + 1)

    for payment, alloc in candidates:
        if payment <= cost:
            # Non-positive unit margin, skip
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
            # Demand exceeds capacity: pick top (v_i - payment) tasks
            per_task_utils = []
            for i in range(Num_MU):
                if alloc[i] > 0:
                    per_task_utils.extend([value[i] - payment] * int(alloc[i]))
            if not per_task_utils:
                continue
            per_task_utils.sort(reverse=True)
            # Only serve the top Num_RB tasks
            served_utils = per_task_utils[:max(0, int(Num_RB))]
            esp = Num_RB * (payment - cost)
            mu = float(np.sum(served_utils))
            if esp > ESP_best:
                ESP_best = esp
                MU_best = mu
                Finish_best = int(Num_RB)

    return ESP_best, MU_best, NI, Finish_best


def main(
    value: Sequence[float],
    payment_min: float,
    payment_max: float,
    cost: float,
    Num_RB: int,
    NumTask: Sequence[int],
) -> Tuple[float, float, int, int]:
    """
    End-to-end ConSpot baseline:
    1) generate candidate (payment, allocation) pairs,
    2) select the best one under the global resource cap.

    Returns:
        (ESP_Utility, MU_Utility, NI, FinishTaskNum)
    """
    candidates = generate_candidates(value, payment_min, payment_max, NumTask)
    return select_best_candidate(candidates, value, cost, Num_RB)

if __name__ == "__main__":
    v = [8.0, 10.0, 12.0, 7.5]
    cost = 6.0
    rb = 10
    maxtask = [3, 5, 4, 2]
    payment_min, payment_max = min(v), max(v)
    print(main(v, payment_min, payment_max, cost, rb, maxtask))
