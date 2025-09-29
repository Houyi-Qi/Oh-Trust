# baselines/random.py
import numpy as np
from typing import Sequence, Tuple

def main(
    value: Sequence[float],
    cost: float,
    Num_RB: int,
    NumTask: Sequence[int],
) -> Tuple[float, float, int, int]:
    """
    Random baseline (rational version):
    - Randomly select a uniform payment price in [cost, max(value)];
    - Each MU (mobile user) randomly decides the number of tasks it wants (0..NumTask[i]);
    - Tasks are allocated in a random order until the resource budget Num_RB is exhausted;
    - Only allocate to MUs with value[i] >= payment price (rational behavior);
    - Return ESP utility, total MU utility, interaction number NI, and total finished tasks.

    Args:
        value: array-like, unit valuation of each MU (length = Num_MU)
        cost: float, unit cost for ESP
        Num_RB: int, total number of available resource blocks
        NumTask: array-like, maximum number of tasks each MU can request

    Returns:
        ESP_Utility: float, utility of the ESP
        MU_Utility: float, total utility of all MUs
        NI: int, interaction number (defined as (Num_MU+1)*2)
        FinishTaskNum: int, total number of finished tasks
    """

    value = np.asarray(value, dtype=float)
    NumTask = np.asarray(NumTask, dtype=int)
    Num_MU = value.shape[0]

    if NumTask.shape[0] != Num_MU:
        raise ValueError("value and NumTask must have the same length")

    # Determine the upper bound for random payment
    payment_max = float(np.max(value)) if Num_MU > 0 else float(cost)
    lo, hi = float(cost), float(payment_max)
    if hi <= lo:
        hi = lo + 1e-6  # avoid invalid range

    # Randomly select a uniform payment price
    payment_random = float(np.random.uniform(lo, hi))

    # Each MU randomly decides its desired number of tasks
    want = np.array([np.random.randint(0, n + 1) for n in NumTask], dtype=int)

    # Random allocation order
    order = np.random.permutation(Num_MU) if Num_MU > 0 else np.array([], dtype=int)
    alloc = np.zeros(Num_MU, dtype=int)
    remain = int(Num_RB)

    # Allocate tasks only if MU's valuation >= payment price
    for i in order:
        if remain <= 0:
            break
        if value[i] >= payment_random:
            give = int(min(want[i], remain))
            alloc[i] = give
            remain -= give

    FinishTaskNum = int(np.sum(alloc))

    # Compute utilities
    ESP_Utility = FinishTaskNum * (payment_random - float(cost))
    MU_Utility = float(np.sum(alloc * (value - payment_random)))

    # Interaction number
    NI = (Num_MU + 1) * 2

    return ESP_Utility, MU_Utility, NI, FinishTaskNum


if __name__ == "__main__":
    v = [8.0, 10.0, 12.0, 7.5]
    cost = 6.0
    rb = 10
    maxtask = [3, 5, 4, 2]
    print(main(v, cost, rb, maxtask))
