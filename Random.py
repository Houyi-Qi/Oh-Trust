import numpy as np
from Utility import counts, values_array, settle_spot
from BiN_TCD import allocate_at_price


def random_spot(values, demand, cost, capacity, prices, *, seed=0):
    d = counts(demand, "demand")
    v = values_array(values, len(d))
    rng = np.random.default_rng(seed)
    eligible_prices = sorted(set(float(p) for p in prices if p >= cost))
    if not eligible_prices or not d.sum() or capacity == 0:
        out = settle_spot(np.zeros_like(d), d, v, 0, cost, capacity)
        out["NI_online"] = 0
        return out
    p = float(rng.choice(eligible_prices))
    # Random offered quantity is a subset of realized demand, never extra demand.
    requested = rng.integers(0, d + 1)
    a = allocate_at_price(v, requested, p, capacity, order=rng.permutation(len(d)))
    out = settle_spot(a, d, v, p, cost, capacity)
    out["NI_online"] = int(np.count_nonzero(d)) + 1
    return out


def main(value, cost, Num_RB, NumTask, *, seed=0, prices=None):
    from BiN_CDO import finite_grid
    if prices is None:
        prices = finite_grid(cost, max(cost, max(value, default=cost)))
    out = random_spot(value, NumTask, cost, Num_RB, prices, seed=seed)
    return dict(method="Random", ESP_Utility=out["seller_utility"],
                MU_Utility=float(out["buyer_utility"].sum()), FinishTaskNum=out["FinishTaskNum"],
                NI=out["NI_online"], NI_online=out["NI_online"], NI_offline=0,
                allocation=out["allocation"])


def run(trace, config, seed=0):
    from OhTrust import evaluate_static
    return evaluate_static("Random", trace, config, seed=seed)


if __name__ == "__main__":
    from OhTrust import cli_main
    cli_main(default_method="Random")
