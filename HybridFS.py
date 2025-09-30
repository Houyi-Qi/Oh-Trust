# HybridFS.py
# ---------------------------------------------------
# HYBRIDFS baseline:
#   Futures (design -> selection -> execution) + Spot top-up
# ---------------------------------------------------
"""
HYBRIDFS Baseline in the Oh-Trust Framework
-------------------------------------------
This baseline first designs futures contracts (grid-search over payment/penalties,
and per-buyer contract size K), selects the best candidate maximizing the ESP's
expected utility under capacity + overbooking constraints, executes the futures
allocation up to the real capacity, and finally uses any remaining capacity in a
spot-stage top-up (simplified ConSpot).

Paper context:
"Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with
 Smart Reputation Update over Dynamic Edge Networks" (IEEE TETC, under review 2025).

Returned schema (fixed across the repository for logging/evaluation):
{
    "method": "HYBRIDFS",
    "ESP_Utility": float(ESP_total),
    "MU_Utility": float(MU_total),
    "FinishTaskNum": int(Finish_total),
    "NI": int(NI_total)
}
"""

from __future__ import annotations
from typing import Sequence, Tuple, List, Dict, Any
import numpy as np

# External modules in your codebase (filenames should be Utility.py, Pr.py)
import Utility
import Pr


# -------------------------
# Inline buyer-side risk constraint (same as ConFutures)
# -------------------------
def risk_FU(expected_utility: float, threshold: float = 0.3) -> bool:
    """
    Buyer-side risk constraint for futures contracts.
    Accept a contract only if expected utility >= threshold (default 0.3).

    Args:
        expected_utility (float): FU's expected utility for a candidate.
        threshold (float): Minimal acceptable expected utility.

    Returns:
        bool: True if acceptable, False otherwise.
    """
    return expected_utility >= threshold


# -------------------------
# (1) PURE FUTURES: candidate generation (α/β/φ used)
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
    Grid-search over (payment, penalty_FU, penalty_ESP).
    For each triple, scan contract size K (max->min), and record each buyer's
    max acceptable size that passes risk_FU on expected utility.

    Scanning ranges (descending over integer grids):
      - payment in [max(payment_Future_min, cost_ESP), floor(payment_Future_max)]
      - penalty_FU in [floor(penalty_FU_min), floor(penalty_FU_max)]
      - penalty_ESP in [floor(penalty_ESP_min), floor(penalty_ESP_max)]
      - K in [FU_TaskNum_min, FU_TaskNum_max]

    Returns:
        Tuple[List[Dict[str, Any]], int]: (candidates, NI)
            candidates[i] = {
                "payment": float,
                "penalty_FU": float,
                "penalty_ESP": float,
                "FU_max_tasks": np.ndarray[int] of shape (Num_FU,)
            }
            NI accumulates interaction counts to match repo style.
    """
    value_FU_OneTask = np.asarray(value_FU_OneTask, dtype=float)
    Num_FU = value_FU_OneTask.shape[0]

    # Payment lower bound should not be below cost (rational contract)
    payment_lb = max(float(payment_Future_min), float(cost_ESP))

    # Descending integer grids to mirror original loop style
    pay_hi = int(np.floor(payment_Future_max))
    pay_lo = int(np.floor(payment_lb))
    pfu_hi = int(np.floor(penalty_FU_max))
    pfu_lo = int(np.floor(penalty_FU_min))
    pesp_hi = int(np.floor(penalty_ESP_max))
    pesp_lo = int(np.floor(penalty_ESP_min))

    candidates: List[Dict[str, Any]] = []
    NI = 0

    for payment in range(pay_hi, pay_lo - 1, -1):
        for penalty_FU in range(pfu_hi, pfu_lo - 1, -1):
            for penalty_ESP in range(pesp_hi, pesp_lo - 1, -1):

                FU_max = np.zeros(Num_FU, dtype=int)

                for K in range(int(FU_TaskNum_max), int(FU_TaskNum_min) - 1, -1):
                    # Probability terms from Pr module
                    Alpha = Pr.Pr_alpha(FU_TaskNum_min, FU_TaskNum_max, K)  # returns (alpha, 1-alpha)
                    Beta  = Pr.pr_beta_1(FU_TaskNum_min, FU_TaskNum_max, Num_FU, ESP_capability)
                    Phi   = Pr.calculate_probabilities(FU_TaskNum_min, FU_TaskNum_max, Num_FU, ESP_capability)

                    # Expected utility per buyer under (payment, penalties, K)
                    U_F_expected = Utility.U_F_i_expected(
                        Alpha[0], Beta,
                        int(FU_TaskNum_min), int(FU_TaskNum_max),
                        value_FU_OneTask, float(payment),
                        (float(payment_Future_min) + float(payment_Future_max)) / 2.0,
                        float(penalty_FU), Phi, float(penalty_ESP), int(K)
                    )
                    # Count interactions (match repo style)
                    NI += (Num_FU + 1)

                    # Risk screen for each buyer
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


def select_best_futures_candidate(
    candidates: List[Dict[str, Any]],
    cost_ESP: float,
    ESP_capability: int,
    FU_TaskNum_min: int,
    FU_TaskNum_max: int,
    OverbookRate: float = 0.2,
) -> Dict[str, Any]:
    """
    Pick feasible candidate with max ESP expected utility.
    Feasible if sum(FU_max_tasks) <= (1 + OverbookRate) * ESP_capability.
    Uses β (capacity-OK probability) and φ (over-demand probability).

    Returns:
        Dict[str, Any]: {..., "ESP_EUtility": float, "feasible": bool}
                        If none feasible, returns {"feasible": False}.
    """
    best: Dict[str, Any] = {"feasible": False}
    best_u = -np.inf
    cap_allow = (1.0 + OverbookRate) * float(ESP_capability)

    for cand in candidates:
        total_max = int(np.sum(cand["FU_max_tasks"]))
        if total_max <= cap_allow:
            # Compute β, φ for the total contracted amount
            beta = Pr.pr_beta_1(int(FU_TaskNum_min), int(FU_TaskNum_max), int(total_max), int(ESP_capability))
            phi  = Pr.calculate_probabilities(int(FU_TaskNum_min), int(FU_TaskNum_max), int(total_max), int(ESP_capability))

            esp_u = Utility.U_ESP_Expected(
                int(total_max),
                int(FU_TaskNum_min), int(FU_TaskNum_max),
                float(cand["payment"]), float(cost_ESP),
                float(cand["penalty_FU"]), float(cand["penalty_ESP"]),
                int(ESP_capability),
                float(beta), float(phi)
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
# (2) FUTURES EXECUTION: settle with actual deliveries (capacity-limited)
# -------------------------
def futures_execution_settlement(
    value: Sequence[float],
    payment_fut: float,
    contract_numTask: Sequence[int],
    capacity_exec: int,
    penalty_FU: float,
    cost: float,
) -> Tuple[float, float, int, np.ndarray]:
    """
    Execute futures: allocate actual deliveries up to capacity_exec.

    Rule:
      Greedily allocate by per-task margin (value - payment_fut) in descending
      order over the contracted tasks, then settle with penalties for any
      under-fulfillment.

    Returns:
        Tuple[float, float, int, np.ndarray]:
            ESP_Utility_fut, MU_Utility_fut, FinishTaskNum_fut, actual_vector (per-buyer).
    """
    value = np.asarray(value, dtype=float)
    contract_numTask = np.asarray(contract_numTask, dtype=int)
    Num_MU = value.shape[0]

    # Build per-task utils only for contracted tasks
    per_task_utils: List[float] = []
    indices: List[int] = []
    for i in range(Num_MU):
        ci = int(contract_numTask[i])
        if ci > 0:
            per_task_utils.extend([value[i] - payment_fut] * ci)
            indices.extend([i] * ci)

    actual = np.zeros(Num_MU, dtype=int)
    if len(per_task_utils) > 0 and capacity_exec > 0:
        order = np.argsort(per_task_utils)[::-1]  # descending
        chosen = order[:min(capacity_exec, len(order))]
        for k in chosen:
            actual[indices[k]] += 1

    # Settle utilities
    ESP_Utility = 0.0
    MU_Utility = 0.0
    for i in range(Num_MU):
        if actual[i] < contract_numTask[i]:
            MU_Utility += actual[i] * (value[i] - payment_fut) \
                          - (contract_numTask[i] - actual[i]) * penalty_FU
            ESP_Utility += actual[i] * (payment_fut - cost) \
                           + (contract_numTask[i] - actual[i]) * penalty_FU
        else:
            MU_Utility += contract_numTask[i] * (value[i] - payment_fut)
            ESP_Utility += contract_numTask[i] * (payment_fut - cost)

    FinishTaskNum = int(np.sum(actual))
    return float(ESP_Utility), float(MU_Utility), FinishTaskNum, actual


# -------------------------
# (3) SPOT TOP-UP: simplified ConSpot on remaining capacity
# -------------------------
def spot_topup_conspot(
    value: Sequence[float],
    NumTask: Sequence[int],
    cost: float,
    payment_min: float,
    payment_max: float,
    Num_RB: int,
) -> Tuple[float, float, int, int]:
    """
    Simplified ConSpot used only for remaining capacity.
    Scan payment from payment_max down to payment_min, form candidate alloc,
    and pick the payment that maximizes ESP utility under Num_RB.

    Returns:
        Tuple[float, float, int, int]:
            ESP_Utility_spot, MU_Utility_spot, NI_spot, FinishTaskNum_spot
    """
    value = np.asarray(value, dtype=float)
    NumTask = np.asarray(NumTask, dtype=int)
    Num_MU = value.shape[0]

    NI = 0
    ESP_best = 0.0
    MU_best = 0.0
    Finish_best = 0

    p = float(payment_max)
    while p >= payment_min:
        alloc = np.zeros(Num_MU, dtype=int)
        for i in range(Num_MU):
            if value[i] >= p:
                alloc[i] = NumTask[i]
        NI += (Num_MU + 1)

        total = int(np.sum(alloc))
        if p > cost and total <= Num_RB:
            esp = total * (p - cost)
            if esp > ESP_best:
                ESP_best = esp
                MU_best = float(np.sum(alloc * (value - p)))
                Finish_best = total
        elif p > cost and total > Num_RB:
            # Pick top (v_i - p) tasks
            util_tasks: List[float] = []
            for i in range(Num_MU):
                if alloc[i] > 0:
                    util_tasks.extend([float(value[i] - p)] * int(alloc[i]))
            if util_tasks:
                util_tasks.sort(reverse=True)
                util_tasks = util_tasks[:Num_RB]
                esp = Num_RB * (p - cost)
                mu = float(np.sum(util_tasks))
                if esp > ESP_best:
                    ESP_best = esp
                    MU_best = mu
                    Finish_best = int(Num_RB)

        p -= 1.0

    return float(ESP_best), float(MU_best), int(NI), int(Finish_best)


# -------------------------
# HYBRIDFS: futures first, then spot for remaining capacity
# -------------------------
def main_hybrid(
    # Futures inputs
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
    # Spot inputs (used only if capacity remains after futures execution)
    value_spot: Sequence[float],
    NumTask_spot: Sequence[int],
    payment_spot_min: float,
    payment_spot_max: float,
) -> Dict[str, float]:
    """
    HYBRIDFS baseline (end-to-end):
      Step 1 (Futures): pick a futures contract set (payment, penalties, per-buyer max).
      Step 2 (Execution): allocate actual deliveries to contracted buyers up to capacity.
      Step 3 (Spot top-up): if capacity remains, run a simplified ConSpot to use the rest.

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
    # ---- Futures: generate and pick best candidate ----
    cand, NI_fut = generate_futures_candidates(
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
        FU_TaskNum_max=FU_TaskNum_max
    )
    best = select_best_futures_candidate(
        candidates=cand,
        cost_ESP=cost_ESP,
        ESP_capability=ESP_capability,
        FU_TaskNum_min=FU_TaskNum_min,
        FU_TaskNum_max=FU_TaskNum_max,
        OverbookRate=0.2
    )

    if not best.get("feasible", False):
        # No feasible futures candidate -> fall back to pure spot on full capacity
        ESP_s, MU_s, NI_s, Fin_s = spot_topup_conspot(
            value=value_spot,
            NumTask=NumTask_spot,
            cost=cost_ESP,
            payment_min=payment_spot_min,
            payment_max=payment_spot_max,
            Num_RB=int(ESP_capability)
        )
        return {
            "method": "ohtrust",
            "ESP_Utility": float(ESP_s),
            "MU_Utility": float(MU_s),
            "FinishTaskNum": int(Fin_s),
            "NI": int(NI_fut + NI_s),
        }

    payment_fut   = float(best["payment"])
    penalty_FU    = float(best["penalty_FU"])
    contract_list = np.asarray(best["FU_max_tasks"], dtype=int)

    # ---- Futures execution: allocate up to capacity ----
    ESP_f, MU_f, Fin_f, _actual_vec = futures_execution_settlement(
        value=np.asarray(value_FU_OneTask, dtype=float),
        payment_fut=payment_fut,
        contract_numTask=contract_list,
        capacity_exec=int(ESP_capability),
        penalty_FU=penalty_FU,
        cost=float(cost_ESP),
    )

    # ---- Remaining capacity goes to spot top-up (if any) ----
    remaining = int(ESP_capability) - int(Fin_f)
    if remaining > 0:
        ESP_s, MU_s, NI_s, Fin_s = spot_topup_conspot(
            value=value_spot,
            NumTask=NumTask_spot,
            cost=cost_ESP,
            payment_min=payment_spot_min,
            payment_max=payment_spot_max,
            Num_RB=remaining
        )
    else:
        ESP_s = MU_s = 0.0
        NI_s = Fin_s = 0

    ESP_total   = float(ESP_f + ESP_s)
    MU_total    = float(MU_f + MU_s)
    NI_total    = int(NI_fut + NI_s)
    Finish_total = int(Fin_f + Fin_s)

    return {
        "method": "ohtrust",
        "ESP_Utility": float(ESP_total),
        "MU_Utility": float(MU_total),
        "FinishTaskNum": int(Finish_total),
        "NI": int(NI_total),
    }


# -------------------------
# Minimal self-test (randomized) to verify runnability
# -------------------------
if __name__ == "__main__":
    # Synthetic quick check; replace with real pipeline in your repo
    np.random.seed(0)

    Num_FU = 10
    ESP_cap = 40

    # Futures-side valuations (per task) for FUs
    v_fu = np.random.uniform(5.0, 15.0, size=Num_FU)

    # Futures search ranges
    pF_min, pF_max = 6.0, 12.0
    qFU_min, qFU_max = 0.5, 3.0
    qESP_min, qESP_max = 0.5, 5.0
    cost_esp = 5.0
    K_min, K_max = 3, 7

    # Spot pool (can be FUs+OUs concatenated in your pipeline)
    Num_spot = 15
    v_spot = np.random.uniform(4.0, 14.0, size=Num_spot)
    n_spot = np.random.randint(1, 5, size=Num_spot)
    pS_min, pS_max = cost_esp, float(np.max(v_spot))

    res = main_hybrid(
        value_FU_OneTask=v_fu,
        payment_Future_min=pF_min,
        payment_Future_max=pF_max,
        penalty_FU_min=qFU_min,
        penalty_FU_max=qFU_max,
        penalty_ESP_min=qESP_min,
        penalty_ESP_max=qESP_max,
        cost_ESP=cost_esp,
        ESP_capability=ESP_cap,
        FU_TaskNum_min=K_min,
        FU_TaskNum_max=K_max,
        value_spot=v_spot,
        NumTask_spot=n_spot,
        payment_spot_min=pS_min,
        payment_spot_max=pS_max,
    )
    print(res)
