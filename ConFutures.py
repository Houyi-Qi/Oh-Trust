# ConFutures.py
# ---------------------------------------------------
# Pure Futures Baseline (ConFutures)
# ---------------------------------------------------
# This module implements the PURE FUTURES baseline used in the
# Oh-Trust paper. It performs a grid search over futures contract
# parameters and selects the candidate maximizing the ESP's
# expected utility under capacity + overbooking constraints.
#
# Paper context:
# "Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling
#  with Smart Reputation Update over Dynamic Edge Networks"
# (IEEE TETC, under review 2025).
#
# The module provides:
#   - risk_FU(): buyer-side (FU) risk acceptance check
#   - generate_futures_candidates(): grid-search contract candidates
#   - select_best_futures_candidate(): pick best candidate by ESP expectation
#   - main_futures(): helper to run the futures phase end-to-end
#   - main_spot(): settlement under realized task completions
#   - main(): top-level wrapper returning a fixed-schema dict for logging
#
# Returned schema for main():
# {
#     "method": "ConFutures",
#     "ESP_Utility": float(ESP_total),
#     "MU_Utility": float(MU_total),
#     "FinishTaskNum": int(Fin_total),
#     "NI": int(NI_total)
# }
# ---------------------------------------------------

from typing import Sequence, Tuple, List, Dict, Any
import numpy as np

import Utility
import Pr


# -------------------------
# Buyer-side risk constraint
# -------------------------
def risk_FU(expected_utility: float, threshold: float = 0.3) -> bool:
    """
    Buyer-side risk constraint for PURE FUTURES (ConFutures).
    A buyer (FU) accepts a futures contract only if its expected
    long-term utility is >= threshold.

    Args:
        expected_utility (float): FU's expected utility under the candidate.
        threshold (float): Minimal acceptable expected utility.

    Returns:
        bool: True if acceptable, False otherwise.
    """
    return expected_utility >= threshold


# -------------------------
# Candidate generation
# -------------------------
def generate_futures_candidates(
    value_FU_OneTask: Sequence[float],
    payment_Future_min: float,
    payment_Future_max: float,
    penalty_FU_min: float,
    penalty_FU_max: float,
    penalty_ESP_min: float,
    penalty_ESP_max: float,
    cost_ESP: float,
    ESP_capability: int,
    FU_TaskNum_min: int,
    FU_TaskNum_max: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Grid-search futures contract candidates.
    Each candidate comprises (payment, penalty_FU, penalty_ESP, FU_max_tasks[i]).

    Scanning ranges:
      - payment in [max(payment_Future_min, cost_ESP), floor(payment_Future_max)]
      - penalty_FU in [floor(penalty_FU_min), floor(penalty_FU_max)]
      - penalty_ESP in [floor(penalty_ESP_min), floor(penalty_ESP_max)]
      - K in [FU_TaskNum_min, FU_TaskNum_max] as a per-FU upper bound

    For each (payment, penalty_FU, penalty_ESP, K):
      1) Compute probability terms (Alpha, Beta, Phi) via Pr.py
      2) Compute per-FU expected utility via Utility.U_F_i_expected(...)
      3) If risk_FU(U_F_expected[i]) holds, update FU_max_tasks[i] = max(FU_max_tasks[i], K)

    Args:
        value_FU_OneTask (Sequence[float]): Per-FU unit valuation.
        payment_Future_min/max (float): Payment scan bounds (inclusive by flooring).
        penalty_FU_min/max (float): FU penalty scan bounds (floored).
        penalty_ESP_min/max (float): ESP penalty scan bounds (floored).
        cost_ESP (float): ESP unit cost.
        ESP_capability (int): ESP capacity (resource blocks).
        FU_TaskNum_min/max (int): Lower/upper bounds for per-FU contract size K.

    Returns:
        Tuple[List[Dict[str, Any]], int]:
            - candidates: list of dicts:
                {
                    "payment": float,
                    "penalty_FU": float,
                    "penalty_ESP": float,
                    "FU_max_tasks": np.ndarray[int] of shape (Num_FU,)
                }
            - NI (int): interaction count accumulated in candidate evaluation.
    """
    value_FU_OneTask = np.asarray(value_FU_OneTask, dtype=float)
    Num_FU = value_FU_OneTask.shape[0]

    # Lower bound of payment should not be below cost
    payment_lb = max(float(payment_Future_min), float(cost_ESP))

    # Integer grids (descending scan)
    pay_hi = int(np.floor(payment_Future_max))
    pay_lo = int(np.floor(payment_lb))
    pfu_hi = int(np.floor(penalty_FU_max))
    pfu_lo = int(np.floor(penalty_FU_min))
    pesp_hi = int(np.floor(penalty_ESP_max))
    pesp_lo = int(np.floor(penalty_ESP_min))

    # Contract size grid
    k_hi = int(FU_TaskNum_max)
    k_lo = int(FU_TaskNum_min)

    candidates: List[Dict[str, Any]] = []
    NI = 0

    for payment in range(pay_hi, pay_lo - 1, -1):
        for penalty_FU in range(pfu_hi, pfu_lo - 1, -1):
            for penalty_ESP in range(pesp_hi, pesp_lo - 1, -1):

                FU_max = np.zeros(Num_FU, dtype=int)

                for K in range(k_hi, k_lo - 1, -1):
                    # Probability terms from Pr module
                    Alpha = Pr.Pr_alpha(k_lo, k_hi, K)
                    Beta = Pr.pr_beta_1(k_lo, k_hi, Num_FU, ESP_capability)
                    Phi = Pr.calculate_probabilities(k_lo, k_hi, Num_FU, ESP_capability)

                    # Expected utility per buyer
                    U_F_expected = Utility.U_F_i_expected(
                        Alpha[0], Beta,
                        k_lo, k_hi,
                        value_FU_OneTask, float(payment),
                        (float(payment_Future_min) + float(payment_Future_max)) / 2.0,
                        float(penalty_FU), Phi, float(penalty_ESP), int(K)
                    )
                    NI += (Num_FU + 1)

                    # Risk screening for each FU
                    for i in range(Num_FU):
                        if risk_FU(U_F_expected[i]):
                            FU_max[i] = max(FU_max[i], int(K))

                candidates.append({
                    "payment": float(payment),
                    "penalty_FU": float(penalty_FU),
                    "penalty_ESP": float(penalty_ESP),
                    "FU_max_tasks": FU_max
                })

    return candidates, int(NI)


# -------------------------
# Best candidate selection
# -------------------------
def select_best_futures_candidate(
    candidates: List[Dict[str, Any]],
    cost_ESP: float,
    ESP_capability: int,
    FU_TaskNum_min: int,
    FU_TaskNum_max: int,
    OverbookRate: float = 0.2,
) -> Dict[str, Any]:
    """
    Choose the candidate maximizing ESP's expected utility,
    under capacity constraint with overbooking allowance.

    Args:
        candidates (List[Dict[str, Any]]): Output of generate_futures_candidates().
        cost_ESP (float): ESP unit cost.
        ESP_capability (int): ESP capacity (RBs).
        FU_TaskNum_min/max (int): Bounds used in expectation computation.
        OverbookRate (float): Allowed overbooking ratio (e.g., 0.2 => 120% of capacity).

    Returns:
        Dict[str, Any]: The best candidate with fields:
            {
                ... original candidate fields ...,
                "ESP_EUtility": float,
                "feasible": bool
            }
            If no feasible candidate, returns {"feasible": False}.
    """
    best: Dict[str, Any] = {"feasible": False}
    best_u = -np.inf

    cap_allow = (1.0 + OverbookRate) * float(ESP_capability)

    for cand in candidates:
        total_max = int(np.sum(cand["FU_max_tasks"]))
        if total_max <= cap_allow:
            # Probabilities for ESP expectation under aggregate load
            beta = Pr.pr_beta_1(FU_TaskNum_min, FU_TaskNum_max, total_max, ESP_capability)
            phi = Pr.calculate_probabilities(FU_TaskNum_min, FU_TaskNum_max, total_max, ESP_capability)

            esp_u = Utility.U_ESP_Expected(
                total_max,
                int(FU_TaskNum_min), int(FU_TaskNum_max),
                float(cand["payment"]), float(cost_ESP),
                float(cand["penalty_FU"]), float(cand["penalty_ESP"]),
                int(ESP_capability),
                beta, phi
            )

            if esp_u > best_u:
                best_u = esp_u
                best = {
                    **cand,
                    "ESP_EUtility": float(esp_u),
                    "feasible": True,
                }

    return best


# -------------------------
# End-to-end PURE FUTURES (helper)
# -------------------------
def main_futures(
    value_FU_OneTask: Sequence[float],
    payment_Future_min: float,
    payment_Future_max: float,
    penalty_FU_min: float,
    penalty_FU_max: float,
    penalty_ESP_min: float,
    penalty_ESP_max: float,
    cost_ESP: float,
    ESP_capability: int,
    FU_TaskNum_min: int,
    FU_TaskNum_max: int,
) -> Tuple[List[Any], int]:
    """
    Run PURE FUTURES search + selection and return the best contract team plus NI.

    Returns:
        Tuple[List[Any], int]:
            - Cteam_best: [payment, penalty_FU, penalty_ESP, FU_max_tasks(list[int])]
            - NI: accumulated interactions in futures candidate evaluation
            If infeasible, returns ([], NI).
    """
    candidates, NI = generate_futures_candidates(
        value_FU_OneTask=value_FU_OneTask,
        payment_Future_min=payment_Future_min,
        payment_Future_max=payment_Future_max,
        penalty_FU_min=penalty_FU_min,
        penalty_FU_max=penalty_FU_max,
        penalty_ESP_min=penalty_ESP_min,
        penalty_ESP_max=penalty_ESP_max,
        cost_ESP=cost_ESP,
        ESP_capability=ESP_capability,
        FU_TaskNum_min=FU_TaskNum_min,
        FU_TaskNum_max=FU_TaskNum_max,
    )

    best = select_best_futures_candidate(
        candidates=candidates,
        cost_ESP=cost_ESP,
        ESP_capability=ESP_capability,
        FU_TaskNum_min=FU_TaskNum_min,
        FU_TaskNum_max=FU_TaskNum_max,
        OverbookRate=0.2,
    )

    if not best.get("feasible", False):
        return [], int(NI)

    Cteam_best = [
        float(best["payment"]),
        float(best["penalty_FU"]),
        float(best["penalty_ESP"]),
        list(map(int, best["FU_max_tasks"])),
    ]
    return Cteam_best, int(NI)


# -------------------------
# Spot-stage settlement
# -------------------------
def main_spot(
    value: Sequence[float],
    payment: float,
    Contract_numTask: Sequence[int],
    numTask_actual: Sequence[int],
    penalty_FU: float,
    cost: float
) -> Tuple[float, float, int, int]:
    """
    Spot-stage settlement given futures contracts (no hybrid).
    Computes realized utilities given actual finished tasks.

    Args:
        value (Sequence[float]): Per-MU unit valuation.
        payment (float): Contract payment (per task).
        Contract_numTask (Sequence[int]): Contracted tasks per MU.
        numTask_actual (Sequence[int]): Actually finished tasks per MU.
        penalty_FU (float): Buyer-side penalty per unfulfilled task.
        cost (float): ESP unit cost.

    Returns:
        Tuple[float, float, int, int]:
            - ESP_Utility (float)
            - MU_Utility (float)
            - FinishTaskNum (int): sum of actual tasks
            - NI_spot (int): interaction count for settlement (~ Num_MU + 1)
    """
    value = np.asarray(value, dtype=float)
    Contract_numTask = np.asarray(Contract_numTask, dtype=int)
    numTask_actual = np.asarray(numTask_actual, dtype=int)

    num_MU = Contract_numTask.shape[0]
    if not (value.shape[0] == num_MU == numTask_actual.shape[0]):
        raise ValueError("value, Contract_numTask, numTask_actual must have the same length")

    ESP_Utility = 0.0
    FU_Utility = np.zeros(num_MU, dtype=float)

    for i in range(num_MU):
        if numTask_actual[i] < Contract_numTask[i]:
            FU_Utility[i] = numTask_actual[i] * (value[i] - payment) \
                            - (Contract_numTask[i] - numTask_actual[i]) * penalty_FU
            ESP_Utility += numTask_actual[i] * (payment - cost) \
                           + (Contract_numTask[i] - numTask_actual[i]) * penalty_FU
        else:
            FU_Utility[i] = Contract_numTask[i] * (value[i] - payment)
            ESP_Utility += Contract_numTask[i] * (payment - cost)

    FinishTaskNum = int(np.sum(numTask_actual))
    NI_spot = (num_MU + 1)  # keep style consistent with other baselines

    return float(ESP_Utility), float(np.sum(FU_Utility)), int(FinishTaskNum), int(NI_spot)


# -------------------------
# Top-level wrapper (fixed-schema output)
# -------------------------
def main(
    # --- futures phase inputs ---
    value_FU_OneTask: Sequence[float],
    payment_Future_min: float,
    payment_Future_max: float,
    penalty_FU_min: float,
    penalty_FU_max: float,
    penalty_ESP_min: float,
    penalty_ESP_max: float,
    cost_ESP: float,
    ESP_capability: int,
    FU_TaskNum_min: int,
    FU_TaskNum_max: int,
    # --- settlement (spot) inputs ---
    value: Sequence[float],
    numTask_actual: Sequence[int],
) -> Dict[str, float]:
    """
    End-to-end PURE FUTURES baseline:
      1) search/select the best futures contract (ConFutures),
      2) settle realized utilities in the spot stage.

    Notes:
      - Contract_numTask is taken from the selected best candidate's FU_max_tasks.
      - The returned dict follows the repo-wide fixed schema.

    Args:
      (See parameter names; `value` and `numTask_actual` are for settlement.)

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
    # Futures search & selection
    Cteam_best, NI_futures = main_futures(
        value_FU_OneTask=value_FU_OneTask,
        payment_Future_min=payment_Future_min,
        payment_Future_max=payment_Future_max,
        penalty_FU_min=penalty_FU_min,
        penalty_FU_max=penalty_FU_max,
        penalty_ESP_min=penalty_ESP_min,
        penalty_ESP_max=penalty_ESP_max,
        cost_ESP=cost_ESP,
        ESP_capability=ESP_capability,
        FU_TaskNum_min=FU_TaskNum_min,
        FU_TaskNum_max=FU_TaskNum_max,
    )

    if not Cteam_best:
        # Infeasible futures => no tasks served
        return {
            "method": "ohtrust",
            "ESP_Utility": float(0.0),
            "MU_Utility": float(0.0),
            "FinishTaskNum": int(0),
            "NI": int(NI_futures),
        }

    payment, penalty_FU, penalty_ESP, FU_max_tasks = Cteam_best
    Contract_numTask = np.asarray(FU_max_tasks, dtype=int)

    # Spot settlement given realized tasks
    ESP_total, MU_total, Fin_total, NI_spot = main_spot(
        value=value,
        payment=float(payment),
        Contract_numTask=Contract_numTask,
        numTask_actual=numTask_actual,
        penalty_FU=float(penalty_FU),
        cost=float(cost_ESP),
    )

    NI_total = int(NI_futures)

    return {
        "method": "ConFutures",
        "ESP_Utility": float(ESP_total),
        "MU_Utility": float(MU_total),
        "FinishTaskNum": int(Fin_total),
        "NI": int(NI_total),
    }


if __name__ == "__main__":
    # Minimal runnable example (toy numbers for sanity check)
    rng = np.random.default_rng(0)

    # Futures inputs
    value_FU_OneTask = [8.0, 9.5, 7.8, 10.2]
    payment_Future_min, payment_Future_max = 6.0, 12.0
    penalty_FU_min, penalty_FU_max = 1.0, 5.0
    penalty_ESP_min, penalty_ESP_max = 1.0, 5.0
    cost_ESP = 6.0
    ESP_capability = 10
    FU_TaskNum_min, FU_TaskNum_max = 1, 5

    # Settlement inputs
    value = value_FU_OneTask  # reuse for simplicity
    # Suppose realized tasks are random up to the candidate's per-FU cap; here, mock with 0..3
    numTask_actual = [int(x) for x in rng.integers(low=0, high=4, size=len(value))]

    res = main(
        value_FU_OneTask=value_FU_OneTask,
        payment_Future_min=payment_Future_min,
        payment_Future_max=payment_Future_max,
        penalty_FU_min=penalty_FU_min,
        penalty_FU_max=penalty_FU_max,
        penalty_ESP_min=penalty_ESP_min,
        penalty_ESP_max=penalty_ESP_max,
        cost_ESP=cost_ESP,
        ESP_capability=ESP_capability,
        FU_TaskNum_min=FU_TaskNum_min,
        FU_TaskNum_max=FU_TaskNum_max,
        value=value,
        numTask_actual=numTask_actual,
    )
    print(res)
