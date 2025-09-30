# BiN_CDO.py
# ----------------------------------------------------
# BiN_CDO: Futures trading (ConFutures) for Oh-Trust
# - Candidate generation (payment, penalties, size K)
# - Best candidate selection under overbooking
# - Futures execution with actual demand + settlement
#   (returns ESP/MU utilities and N+/N- for reputation)
# ----------------------------------------------------
"""
BiN_CDO: Futures Trading Component for the Oh-Trust Framework
-------------------------------------------------------------
This module provides the futures-side mechanics used by Oh-Trust:

1) Candidate Generation
   - Grid-search over payment and penalty parameters (payment, penalty_FU, penalty_ESP),
     and scan a per-buyer contract size K from K_max down to K_min.
   - For each buyer (FU), record the maximum acceptable K that passes a risk check
     (based on expected utility).

2) Best Candidate Selection
   - Among generated candidates, select the feasible one (respecting overbooking)
     that maximizes the ESP's expected utility.
   - A candidate is feasible if the sum of per-buyer maxima does not exceed
     (1 + OverbookRate) * ESP_capability.

3) Futures Execution with Actual Demand
   - Given realized per-buyer demand, greedily allocate up to the minimum of
     (contracted, demand) under the capacity limit, by descending per-task margin
     (v_i - payment_fut).
   - Settle utilities and compute N_plus/N_minus for reputation updates.

Notes
-----
- The code here focuses on clarity and reproducibility of the baseline; performance
  optimizations are out of scope.
"""

from __future__ import annotations
from typing import Sequence, Tuple, List, Dict, Any
import numpy as np
import math

# Use your existing modules
import Pr
import Utility


def risk_FU(expected_utility: float, threshold: float = 0.0) -> bool:
    """Buyer-side risk check: accept if expected utility >= threshold."""
    return expected_utility >= threshold


def generate_futures_candidates(
    value_FU_OneTask: Sequence[float],
    payment_min: float, payment_max: float,
    penalty_FU_min: float, penalty_FU_max: float,
    penalty_ESP_min: float, penalty_ESP_max: float,
    cost_ESP: float, ESP_capability: int,
    K_min: int, K_max: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Generate futures candidates via grid search.

    For each triple (payment, penalty_FU, penalty_ESP) and a contract size K
    scanned from K_max down to K_min, evaluate each buyer's expected utility
    and record the maximum acceptable K passing the risk constraint.

    Args:
        value_FU_OneTask: Per-buyer unit valuation (length = Num_FU).
        payment_min/payment_max: Bounds for payment scanning (floored to ints).
        penalty_FU_min/penalty_FU_max: Buyer-side penalty bounds (floored).
        penalty_ESP_min/penalty_ESP_max: ESP-side penalty bounds (floored).
        cost_ESP: ESP unit cost.
        ESP_capability: ESP capacity (resource blocks).
        K_min/K_max: Lower/upper bounds for per-buyer contract size.

    Returns:
        candidates: List of candidate dicts with keys:
            {
              "payment": float,
              "penalty_FU": float,
              "penalty_ESP": float,
              "FU_max_tasks": np.ndarray[int] of shape (Num_FU,)
            }
        NI: Interaction count accumulated during evaluation (for logging parity).
    """
    v = np.asarray(value_FU_OneTask, dtype=float)
    Num_FU = v.shape[0]

    # Lower bound for payment scanning; follows the original code path
    payment_lb = min(float(payment_min), float(cost_ESP))

    candidates: List[Dict[str, Any]] = []
    NI = 0

    for p in range(int(math.floor(payment_max)), int(math.floor(payment_lb)) - 1, -1):
        for q_f in range(int(math.floor(penalty_FU_max)), int(math.floor(penalty_FU_min)) - 1, -1):
            for q_e in range(int(math.floor(penalty_ESP_max)), int(math.floor(penalty_ESP_min)) - 1, -1):

                FU_max = np.zeros(Num_FU, dtype=int)

                for K in range(int(K_max), int(K_min) - 1, -1):
                    # Probability terms from Pr module
                    Alpha = Pr.Pr_alpha(K_min, K_max, K)   # returns (A0, A1)
                    Beta  = Pr.pr_beta_1(K_min, K_max, Num_FU, ESP_capability)
                    Phi   = Pr.calculate_probabilities(K_min, K_max, Num_FU, ESP_capability)

                    # Expected utility for each buyer under size K
                    # (U_F_i_expected accepts scalar v_i; compute per buyer)
                    for i in range(Num_FU):
                        Uexp = Utility.U_F_i_expected(
                            Alpha[0], Beta,
                            K_min, K_max,
                            float(v[i]),
                            float(p),
                            (payment_min + payment_max) / 2.0,
                            float(q_f),
                            Phi,
                            float(q_e),
                            int(K),
                        )
                        NI += (Num_FU + 1)
                        if risk_FU(Uexp):
                            FU_max[i] = max(FU_max[i], int(K))

                candidates.append({
                    "payment": float(p),
                    "penalty_FU": float(q_f),
                    "penalty_ESP": float(q_e),
                    "FU_max_tasks": FU_max,
                })

    return candidates, NI


def select_best_futures_candidate(
    candidates: List[Dict[str, Any]],
    cost_ESP: float, ESP_capability: int,
    K_min: int, K_max: int,
    OverbookRate: float = 0.2,
) -> Dict[str, Any]:
    """
    Select the feasible candidate that maximizes the ESP's expected utility.

    Feasibility:
        sum(FU_max_tasks) <= (1 + OverbookRate) * ESP_capability

    Args:
        candidates: Output of generate_futures_candidates().
        cost_ESP: ESP unit cost.
        ESP_capability: ESP capacity (resource blocks).
        K_min/K_max: Bounds used in expectation computation.
        OverbookRate: Allowed overbooking ratio (e.g., 0.2 => 120% capacity).

    Returns:
        Best candidate dict with extra fields:
            {
              ...original candidate fields...,
              "ESP_EUtility": float,
              "feasible": True
            }
        If none feasible, returns {"feasible": False}.
    """
    best = {"feasible": False}
    best_u = -np.inf
    cap_allow = (1.0 + OverbookRate) * float(ESP_capability)

    for cand in candidates:
        total_max = int(np.sum(cand["FU_max_tasks"]))
        if total_max <= cap_allow:
            # Utility.U_ESP_Expected requires beta and phi
            beta = Pr.pr_beta_1(K_min, K_max, total_max, ESP_capability)
            phi  = Pr.calculate_probabilities(K_min, K_max, total_max, ESP_capability)

            esp_u = Utility.U_ESP_Expected(
                total_max,
                int(K_min), int(K_max),
                float(cand["payment"]), float(cost_ESP),
                float(cand["penalty_FU"]), float(cand["penalty_ESP"]),
                int(ESP_capability),
                beta, phi,
            )
            if esp_u > best_u:
                best_u = esp_u
                best = {**cand, "ESP_EUtility": float(esp_u), "feasible": True}
    return best


def execute_with_demand(
    value: Sequence[float],
    payment_fut: float,
    contract_numTask: Sequence[int],
    actual_demand: Sequence[int],
    capacity_exec: int,
    penalty_FU: float,
    cost: float,
) -> Tuple[float, float, int, np.ndarray, int, int]:
    """
    Execute futures with realized demand and settle utilities.

    Allocation rule:
        - For each buyer i, potential deliverable = min(contract_i, demand_i).
        - Build per-task margins (v_i - payment_fut) for these deliverables.
        - Greedily allocate tasks by descending margin under capacity_exec.

    Settlement:
        - If delivered < contracted: FU pays penalty_FU per shortfall;
          ESP receives the same penalty.
        - Otherwise: utilities are computed on the contracted amount.

    Returns:
        (ESP_Utility, MU_Utility, FinishTaskNum, actual_alloc_vector, N_plus, N_minus)
        where:
          - FinishTaskNum = sum of actually delivered tasks across buyers
          - N_plus: number of fulfilled contract tasks (for reputation)
          - N_minus: number of defaulted contract tasks (for reputation)
    """
    v = np.asarray(value, dtype=float)
    c = np.asarray(contract_numTask, dtype=int)
    d = np.asarray(actual_demand, dtype=int)
    Num_MU = v.shape[0]

    per_task_util, idx = [], []
    for i in range(Num_MU):
        cap = int(min(c[i], d[i]))
        if cap > 0:
            per_task_util.extend([v[i] - payment_fut] * cap)
            idx.extend([i] * cap)

    alloc = np.zeros(Num_MU, dtype=int)
    if per_task_util and capacity_exec > 0:
        order = np.argsort(per_task_util)[::-1]
        chosen = order[:min(capacity_exec, len(order))]
        for k in chosen:
            alloc[idx[k]] += 1

    ESP_U, MU_U = 0.0, 0.0
    N_plus, N_minus = 0, 0
    for i in range(Num_MU):
        delivered, contracted = int(alloc[i]), int(c[i])
        fulfilled = min(delivered, contracted)
        defaulted = max(contracted - delivered, 0)
        N_plus += fulfilled
        N_minus += defaulted
        if delivered < contracted:
            MU_U += delivered * (v[i] - payment_fut) - (contracted - delivered) * penalty_FU
            ESP_U += delivered * (payment_fut - cost) + (contracted - delivered) * penalty_FU
        else:
            MU_U += contracted * (v[i] - payment_fut)
            ESP_U += contracted * (payment_fut - cost)

    return float(ESP_U), float(MU_U), int(np.sum(alloc)), alloc, int(N_plus), int(N_minus)
