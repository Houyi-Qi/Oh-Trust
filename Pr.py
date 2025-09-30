# Pr.py
# ------------------------------------------------------------
# Probability helpers used by Oh-Trust.
#
# IMPORTANT ABOUT alpha_i:
# - Minimal runnable version (for quick reproducibility, no dataset):
#     Pr_alpha(mu_min, mu_max, K)
#   This treats r_i as a discrete UNIFORM on {mu_min..mu_max} and returns
#     E[alpha_i] = P(r_i > K)  (uniform-based surrogate).
#   This is ONLY for minimal implementation so the system can run without
#   a historical dataset.
#
# - Data-driven version (paper-faithful, Eq.(23)):
#     alpha_from_history(hist_data, n_i)  ->  (# {r > n_i}) / (# {r})
#     alpha_batch_from_history(list_of_arrays, n_vec)
#   Use these when you have real historical {r_i} samples (recommended).
#
# Other functions:
# - pr_beta_1(...) : Normal-approx CDF for sum of n i.i.d. discrete uniforms
#                    U{mu1..mu2}; returns P(sum <= G_E) by default.
# - calculate_probabilities(...) : exact DP for P(sum > G_E) with the same
#                    discrete uniform assumption. Use when n/range are moderate.
# ------------------------------------------------------------

from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple
import math
import numpy as np
from scipy.stats import norm


# ---------------------------
# MINIMAL (uniform-surrogate) alpha_i
# ---------------------------
def Pr_alpha(mu_min: int, mu_max: int, K: int) -> Tuple[float, float]:
    """
    Minimal/surrogate alpha_i for when NO historical data are available.

    Assumption:
        r_i ~ Uniform{mu_min, mu_min+1, ..., mu_max} (discrete uniform)
    We return:
        alpha0 = P(r_i > K)
        alpha1 = 1 - alpha0

    This lets the pipeline run end-to-end even without a dataset.
    For paper-faithful evaluation with real data, prefer the functions
    `alpha_from_history` or `alpha_batch_from_history` below.

    Args:
        mu_min, mu_max : integer bounds of the discrete uniform
        K              : threshold (contract size)

    Returns:
        (alpha0, alpha1) where alpha0 = P(r_i > K).
    """
    mu_min = int(mu_min)
    mu_max = int(mu_max)
    K = int(K)
    if mu_min > mu_max:
        mu_min, mu_max = mu_max, mu_min

    # All mass outside or degenerate
    if mu_min == mu_max:
        alpha0 = 1.0 if mu_max > K else 0.0
        return float(alpha0), float(1.0 - alpha0)

    # Discrete uniform on [mu_min..mu_max]
    denom = (mu_max - mu_min + 1)
    if K < mu_min:
        alpha0 = 1.0
    elif K >= mu_max:
        alpha0 = 0.0
    else:
        # Count of integers strictly greater than K
        greater = (mu_max - max(K, mu_min))  # e.g., K=3, max=5 -> {4,5} -> 2
        alpha0 = greater / denom

    alpha0 = max(0.0, min(1.0, float(alpha0)))
    alpha1 = 1.0 - alpha0
    return alpha0, alpha1


# ---------------------------
# DATA-DRIVEN alpha_i (paper Eq.(23))
# ---------------------------
def alpha_from_history(hist_data: Sequence[float], n_i: int) -> Tuple[float, float]:
    """
    Paper-faithful alpha_i using historical samples (Eq. (23)):

        E[alpha_i] = (# of samples with r_i > n_i) / (# of samples)

    This should be used when you have real historical demand data for buyer i.

    Args:
        hist_data : array-like of historical r_i samples (len = X_i^all)
        n_i       : threshold (e.g., contract size) to test against

    Returns:
        (alpha0, alpha1) where alpha0 = E[alpha_i], alpha1 = 1 - alpha0.
    """
    h = np.asarray(hist_data)
    total = int(h.size)
    if total <= 0:
        # No data -> fall back to a neutral default (0.0, 1.0).
        return 0.0, 1.0

    greater = int(np.sum(h > n_i))  # X_i^1
    alpha0 = greater / total        # Eq.(23)
    alpha0 = max(0.0, min(1.0, float(alpha0)))
    alpha1 = 1.0 - alpha0
    return alpha0, alpha1


def alpha_batch_from_history(
    histories: List[Sequence[float]],
    n_vec: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch version for multiple buyers i = 1..N:

        histories[i] : historical samples for buyer i   (array-like)
        n_vec[i]     : threshold n_i for buyer i        (int)

    Returns:
        alpha0_arr, alpha1_arr  (both shape = (N,))
        where alpha0_arr[i] = E[alpha_i] from Eq.(23)

    This makes it easy to drop into vectorized pipelines.
    """
    N = len(histories)
    if len(n_vec) != N:
        raise ValueError("histories and n_vec must have the same length")

    a0 = np.zeros(N, dtype=float)
    a1 = np.zeros(N, dtype=float)
    for i in range(N):
        alpha0, alpha1 = alpha_from_history(histories[i], int(n_vec[i]))
        a0[i] = alpha0
        a1[i] = alpha1
    return a0, a1


# ---------------------------
# Beta: capacity adequacy probability (normal approx)
# ---------------------------
def pr_beta_1(mu1: int, mu2: int, n: int, G_E: int, return_le: bool = True) -> float:
    """
    Normal approximation for the sum of n i.i.d. discrete uniforms U{mu1..mu2}.
    Interpreted in Oh-Trust as a capacity adequacy probability.

    Returns:
        P(sum <= G_E) if return_le=True (default),
        else P(sum > G_E).

    Notes:
        This remains a distributional assumption for tractability. If you have
        empirical sums, you can replace this by an empirical CDF.
    """
    mu1, mu2, n, G_E = int(mu1), int(mu2), int(n), int(G_E)
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1

    # mean/variance of discrete uniform
    E_d = (mu1 + mu2) / 2.0
    Var_d = ((mu2 - mu1 + 1) ** 2 - 1) / 12.0

    E_sum = n * E_d
    Var_sum = n * Var_d

    if Var_sum <= 0:
        p_le = 1.0 if (n * mu1) <= G_E else 0.0
        return p_le if return_le else (1.0 - p_le)

    z = (G_E - E_sum) / math.sqrt(Var_sum)
    p_le = float(norm.cdf(z))
    return p_le if return_le else (1.0 - p_le)


# ---------------------------
# Phi: exact DP for tail probability
# ---------------------------
def calculate_probabilities(mu1: int, mu2: int, n: int, G_E: int) -> float:
    """
    Exact probability P(sum > G_E) where each variable ~ Uniform{mu1..mu2}.
    Dynamic programming; suitable for moderate n/range only.

    Args:
        mu1, mu2 : bounds of the discrete uniform
        n        : number of variables
        G_E      : threshold to exceed

    Returns:
        pr : float in [0,1], the probability that the sum exceeds G_E
    """
    mu1, mu2, n, G_E = int(mu1), int(mu2), int(n), int(G_E)
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1

    values = np.arange(mu1, mu2 + 1, dtype=int)
    if n <= 0 or values.size == 0:
        return 0.0

    min_sum = n * mu1
    max_sum = n * mu2
    if G_E >= max_sum:
        return 0.0
    if G_E < min_sum:
        return 1.0

    dp = np.zeros((n + 1, max_sum + 1), dtype=float)
    dp[0, 0] = 1.0
    V = float(values.size)

    # Convolution over discrete uniform
    for i in range(1, n + 1):
        prev = dp[i - 1]
        cur = dp[i]
        # For each value v, shift-add prev
        for v in values:
            cur[v:] += prev[:-v]
        cur /= V  # uniform averaging

    pr = float(np.sum(dp[n, G_E + 1:]))
    # numerical guards
    if pr < 0.0:
        pr = 0.0
    if pr > 1.0:
        pr = 1.0
    return pr


# ---------------------------
# Optional quick self-test
# ---------------------------
if __name__ == "__main__":
    # Minimal/surrogate alpha
    print("Pr_alpha (uniform surrogate):", Pr_alpha(1, 5, 3))  # ~ P(r>3) on {1..5} => 2/5

    # Data-driven alpha (Eq.23)
    hist = [2, 3, 4, 5]   # two samples > 3
    print("alpha_from_history:", alpha_from_history(hist, 3))  # -> (2/4, 2/4) = (0.5, 0.5)

    # Beta (normal approx) and Phi (exact DP)
    print("pr_beta_1 P(sum<=600):", pr_beta_1(7, 12, n=100, G_E=600))
    print("calculate_probabilities P(sum>600):", calculate_probabilities(7, 12, n=30, G_E=300))
