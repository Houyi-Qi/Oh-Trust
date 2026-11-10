"""Algorithm 2 adaptation: futures first, then residual FB and arrived OB demand."""
import numpy as np
from Utility import (counts, values_array, capacity_value, nonnegative,
                         execute_futures, settle_spot, combine_market)
from BiN_CDO import finite_grid


def allocate_at_price(values, demand, price, capacity, *, order=None):
    d = counts(demand, "demand")
    v = values_array(values, len(d))
    remaining = capacity_value(capacity)
    allocation = np.zeros_like(d)
    if order is None:
        order = np.argsort(-v, kind="stable")
    for i in order:
        if v[i] >= price:
            allocation[i] = min(int(d[i]), remaining)
            remaining -= int(allocation[i])
    return allocation


def clear_spot(values, residual_demand, cost, capacity, prices):
    d = counts(residual_demand, "residual demand")
    v = values_array(values, len(d))
    cap, c = capacity_value(capacity), nonnegative(cost, "cost")
    best = settle_spot(np.zeros_like(d), d, v, 0.0, c, cap)
    ni = 0
    if cap > 0 and d.sum() > 0:
        for p in sorted(set(prices), reverse=True):
            nonnegative(p, "spot price")
            if p < c:
                continue
            ni += int(np.count_nonzero(d)) + 1
            a = allocate_at_price(v, d, p, cap)
            out = settle_spot(a, d, v, p, c, cap)
            if (out["seller_utility"], out["FinishTaskNum"]) > (best["seller_utility"], best["FinishTaskNum"]):
                best = out
    best["NI_online"] = ni
    return best


def execute_market(contract, demand, values, num_fb, capacity, cost, spot_prices, *,
                   seed=0, spot_enabled=True, spot_limits=()):
    d = counts(demand, "demand")
    v = values_array(values, len(d))
    if not 0 <= num_fb <= len(d) or len(contract.quantities) != num_fb:
        raise ValueError("FB identities must match the contract")
    futures = execute_futures(contract, d[:num_fb], v[:num_fb], capacity, cost, seed=seed)
    remaining = capacity_value(capacity) - futures["FinishTaskNum"]
    residual = d.copy()
    residual[:num_fb] -= futures["allocation"]
    if len(spot_limits):
        limits = counts(spot_limits, "spot limits")
        if limits.shape != residual.shape:
            raise ValueError("spot limits must match buyer identities")
        residual = np.minimum(residual, limits)
    # Volunteers occur only when remaining == 0; they cannot be resold service.
    spot = clear_spot(v, residual, cost, remaining, spot_prices if spot_enabled else ())
    return combine_market(d, v, capacity, cost, futures, spot, num_fb, NI_online=spot["NI_online"])


def conspot(value, NumTask, cost, payment_min, payment_max, Num_RB):
    """Compatibility tuple: seller utility, buyer utility, NI, completed tasks."""
    out = clear_spot(value, NumTask, cost, Num_RB, finite_grid(payment_min, payment_max))
    return (out["seller_utility"], float(out["buyer_utility"].sum()),
            out["NI_online"], out["FinishTaskNum"])
