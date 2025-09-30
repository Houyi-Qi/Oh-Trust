# baselines/random.py
"""
Random Baseline for Resource Allocation in Oh-Trust Framework
-------------------------------------------------------------
This module implements a rational random baseline to compare against
the Oh-Trust mechanism. It simulates task allocation under the following rules:

1. A random uniform payment price is selected in [cost, max(value)].
2. Each MU (mobile user) randomly decides the number of tasks to request (0..NumTask[i]).
3. Tasks are allocated in a random order until the ESP’s resource budget (Num_RB) is exhausted.
4. Only MUs with valuation >= payment price are eligible for allocation (rational behavior).
5. The outcome includes ESP utility, MU utility, interaction number (NI), and finished tasks.

This baseline is provided for reproducibility in the Oh-Trust paper:
"Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update"
(IEEE TETC, under review 2025).
"""

import numpy as np
from typing import Sequence, Tuple, Dict


def main(
    value: Sequence[float],
    cost: float,
    Num_RB: int,
    NumTask: Sequence[int],
) -> Dict[str, float]:
    """
    Run the Random baseline with rational allocation rules.

    Args:
        value (Sequence[float]): Unit valuation of each MU (length = Num_MU).
        cost (float): Unit cost for the ESP.
        Num_RB (int): Total number of available resource blocks.
        NumTask (Sequence[int]): Maximum number of tasks each MU can request.

    Returns:
        dict: A dictionary containing the evaluation results with fixed schema:
            {
                "method": "Random",
                "ESP_Utility": float,
                "MU_Utility": float,
                "FinishTaskNum": int,
                "NI": int
            }
    """
    value = np.asarray(value, dtype=float)
    NumTask = np.asarray(NumTask, dtype=int)
    Num_MU = value.shape[0]

    if NumTask.shape[0] != Num_MU:
        raise ValueError("value and NumTask must have the same length")

    # Determine random payment price range
    payment_max = float(np.max(value)) if Num_MU > 0 else float(cost)
    lo, hi = float(cost), float(payment_max)
    if hi <= lo:
        hi = lo + 1e-6  # avoid invalid range

    payment_random = float(np.random.uniform(lo, hi))

    # Each MU randomly decides its desired number of tasks
    want = np.array([np.random.randint(0, n + 1) for n in NumTask], dtype=int)

    # Random allocation order
    order = np.random.permutation(Num_MU) if Num_MU > 0 else np.array([], dtype=int)
    alloc = np.zeros(Num_MU, dtype=int)
    remain = int(Num_RB)

    # Allocation: only to MUs with value >= payment price
    for i in order:
        if remain <= 0:
            break
        if value[i] >= payment_random:
            give = int(min(want[i], remain))
            alloc[i] = give
            remain -= give

    # Final results
    FinishTaskNum = int(np.sum(alloc))
    ESP_total = FinishTaskNum * (payment_random - float(cost))
    MU_total = float(np.sum(alloc * (value - payment_random)))
    NI_total = (Num_MU + 1) * 2

    return {
        "method": "Random",
        "ESP_Utility": float(ESP_total),
        "MU_Utility": float(MU_total),
        "FinishTaskNum": int(FinishTaskNum),
        "NI": int(NI_total),
    }


if __name__ == "__main__":
    # Example run with toy parameters
    v = [8.0, 10.0, 12.0, 7.5]
    cost = 6.0
    rb = 10
    maxtask = [3, 5, 4, 2]
    result = main(v, cost, rb, maxtask)
    print(result)
