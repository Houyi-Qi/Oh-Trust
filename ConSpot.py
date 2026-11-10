import numpy as np
from Utility import counts, values_array
from BiN_CDO import finite_grid
from BiN_TCD import clear_spot, conspot


def generate_candidates(value, payment_min, payment_max, NumTask):
    demand = counts(NumTask, "demand")
    values = values_array(value, len(demand))
    return [(p, np.where(values >= p, demand, 0)) for p in finite_grid(payment_min, payment_max)]


def select_best_candidate(candidates, value, cost, Num_RB):
    best, ni = (0., 0., 0, 0), 0
    for price, demand in candidates:
        out = clear_spot(value, demand, cost, Num_RB, (price,))
        ni += out["NI_online"]
        if (out["seller_utility"], out["FinishTaskNum"]) > (best[0], best[3]):
            best = out["seller_utility"], float(out["buyer_utility"].sum()), 0, out["FinishTaskNum"]
    return best[0], best[1], ni, best[3]


def main(value, payment_min, payment_max, cost, Num_RB, NumTask):
    out = clear_spot(value, NumTask, cost, Num_RB, finite_grid(payment_min, payment_max))
    return dict(method="ConSpot", ESP_Utility=out["seller_utility"],
                MU_Utility=float(out["buyer_utility"].sum()), FinishTaskNum=out["FinishTaskNum"],
                NI=out["NI_online"], NI_online=out["NI_online"], NI_offline=0,
                allocation=out["allocation"])


def run(trace, config, seed=0):
    from OhTrust import evaluate_static
    return evaluate_static("ConSpot", trace, config, seed=seed)


if __name__ == "__main__":
    from OhTrust import cli_main
    cli_main(default_method="ConSpot")
