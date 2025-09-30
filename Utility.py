# utility.py
# ---------------------------------------------------
# Utility functions for ESP / FU / OU under futures-spot trading
# Strictly consistent with paper equations (Oh-Trust / ConFutures)
# ---------------------------------------------------

import numpy as np
import math


def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Probability of exactly k successes in n Bernoulli trials
    with success probability p.
    """
    if k < 0 or k > n:
        return 0.0
    combination = math.comb(n, k)
    return combination * (p ** k) * ((1 - p) ** (n - k))


# ---------------------------------------------------
# ESP Utility (actual execution, not expectation)
# ---------------------------------------------------
def U_ESP(num_futures, num_spot, payment_futures, payment_spot,
          q_ESP, q_FU, cost, num_volunteer, num_FU_break) -> float:
    """
    Utility of ESP = revenues from futures & spot
                   - penalties paid to volunteers
                   + penalties collected from FUs
    """
    part1 = num_futures * (payment_futures - cost) + num_spot * (payment_spot - cost)
    part2 = q_ESP * num_volunteer
    part3 = q_FU * num_FU_break
    return part1 - part2 + part3


# ---------------------------------------------------
# OU Utility
# ---------------------------------------------------
def Utility_OU(num: int, payment_spot: float, value: float) -> float:
    """
    Utility of OU tasks in spot trading.
    """
    return num * (value - payment_spot)


# ---------------------------------------------------
# FU Utility (actual execution)
# ---------------------------------------------------
def U_F_i(num_task_futures, num_task_spot, num_volunteer, num_task_break,
          payment_futures, payment_spot, value, q_ESP, q_FU) -> float:
    """
    Utility of a single FU given its executed tasks.
    """
    return (num_task_futures * (value - payment_futures)
            + num_task_spot * (value - payment_spot)
            + num_volunteer * q_ESP
            - num_task_break * q_FU)


# ---------------------------------------------------
# FU Expected Utility (paper Eq. with α, β, φ)
# ---------------------------------------------------
def U_F_i_expected(Pr_alpha_i, Pr_beta, NumMin, NumMax, v_i,
                   p_F, p_S, q_F_E, phi_i, q_F_E_to_F, N_i) -> float:
    """
    Expected utility of FU i under futures trading.
    - Pr_alpha_i: α (contract acceptance probability)
    - Pr_beta: β (prob. demand ≤ ESP capacity)
    - phi_i: φ (prob. of over-demand / FU breach)
    """
    # Part 1: FU accepts, capacity sufficient
    part1 = Pr_alpha_i * Pr_beta * (
        N_i * (v_i - p_F)
        + ((NumMax - N_i + 1) / 2 - N_i) * (v_i - p_S)
    )

    # Part 2: FU rejects, capacity sufficient
    part2 = (1 - Pr_alpha_i) * Pr_beta * (
        (N_i - NumMin + 1) / 2 * (v_i - p_F)
        - (N_i - (N_i - NumMin + 1) / 2) * q_F_E
    )

    # Part 3: FU accepts, but capacity exceeded
    part3 = phi_i * (1 - Pr_beta) * (
        (N_i - NumMin + 1) / 2 * (v_i - p_F)
        + (N_i - NumMin + 1) / 2 * q_F_E_to_F
    )

    # Part 4: FU rejects, capacity exceeded, others breach
    part4 = (1 - Pr_alpha_i) * (1 - Pr_beta) * (1 - phi_i) * (
        (N_i - NumMin + 1) / 2 * (v_i - p_F)
        - (N_i - 2 * (N_i - NumMin + 1) / 2) * q_F_E
    )

    return part1 + part2 + part3 + part4


# ---------------------------------------------------
# ESP Expected Utility
# ---------------------------------------------------
def U_ESP_Expected(
    TotalContract: int,
    NumMin: int, NumMax: int,
    p_F: float, c_F: float,
    q_F_E: float, q_E_to_F: float,
    ESP_cap: int,
    beta: float,  # from Pr.pr_beta_1(..., return_le=True)
    phi: float    # from Pr.calculate_probabilities(...)
) -> float:
    """
    Expected utility of ESP under futures trading.
    Strictly paper-aligned: weighted by β (capacity ok) and φ (capacity exceeded).
    """
    # Case 1: demand ≤ ESP capacity (prob = beta)
    served_when_ok = min(TotalContract, ESP_cap) * (p_F - c_F)

    # Case 2: demand > ESP capacity (prob = phi)
    over = max(0, TotalContract - ESP_cap)
    served_when_over = ESP_cap * (p_F - c_F) - over * q_E_to_F

    return beta * served_when_ok + phi * served_when_over
