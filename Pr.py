import hashlib
import numpy as np
from scipy.signal import convolve
from Utility import counts, capacity_value


def Pr_alpha(mu_min, mu_max, K):
    lo, hi = sorted(counts([mu_min, mu_max], "uniform bounds"))
    p = float(np.mean(np.arange(lo, hi + 1) > K))
    return p, 1.0 - p


def alpha_from_history(hist_data, n_i):
    h = counts(hist_data, "history")
    if not len(h):
        raise ValueError("history is empty; supply an explicit synthetic prior")
    p = float(np.mean(h > n_i))
    return p, 1.0 - p


def alpha_batch_from_history(histories, n_vec):
    if len(histories) != len(n_vec):
        raise ValueError("histories and n_vec must have the same length")
    p = np.array([alpha_from_history(h, n)[0] for h, n in zip(histories, n_vec)])
    return p, 1.0 - p


def calculate_probabilities(mu1, mu2, n, G_E):
    lo, hi = sorted(counts([mu1, mu2], "uniform bounds"))
    buyers = capacity_value(n)
    if G_E >= buyers * hi:
        return 0.0
    if G_E < buyers * lo:
        return 1.0
    pmf = np.ones(hi - lo + 1) / (hi - lo + 1)
    dist = np.array([1.0])
    for _ in range(buyers):
        dist = convolve(dist, pmf, method="direct")
    cut = int(np.floor(G_E - buyers * lo)) + 1
    return float(np.clip(dist[max(cut, 0):].sum(), 0.0, 1.0))


def pr_beta_1(mu1, mu2, n, G_E, return_le=True):
    p_over = calculate_probabilities(mu1, mu2, n, G_E)
    return 1.0 - p_over if return_le else p_over


def sample_demand_scenarios(*, history=None, lower=None, upper=None, samples=128, seed=0):
    """Bootstrap complete rows (preserve dependence), or discrete uniforms."""
    if samples < 1 or int(samples) != samples:
        raise ValueError("samples must be a positive integer")
    rng = np.random.default_rng(seed)
    if history is not None:
        h = counts(history, "history", ndim=2)
        if len(h) == 0:
            raise ValueError("history cannot be empty")
        return h[rng.integers(len(h), size=samples)].copy()
    if lower is None or upper is None:
        raise ValueError("provide either history or explicit synthetic bounds")
    lo, hi = counts(lower, "lower"), counts(upper, "upper")
    if lo.shape != hi.shape or np.any(lo > hi):
        raise ValueError("invalid demand bounds")
    return rng.integers(lo, hi + 1, size=(samples, len(lo)))


def capacity_probabilities(contract, demand_scenarios, capacity):
    n, d = counts(contract, "contract"), counts(demand_scenarios, "scenarios", ndim=2)
    cap = capacity_value(capacity)
    if not len(d) or d.shape[1] != len(n):
        raise ValueError("scenarios must be nonempty and match contract vector")
    p_over = float(np.mean(np.minimum(n, d).sum(axis=1) > cap))
    return {"p_over": p_over, "p_sufficient": 1.0 - p_over}


def scenario_digest(scenarios, priorities):
    h = hashlib.sha256()
    for arr in (scenarios, priorities):
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()
