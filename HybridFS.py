import numpy as np
from ConFutures import generate_futures_candidates, select_best_futures_candidate, main_futures
from BiN_CDO import execute_with_demand, finite_grid
from BiN_TCD import conspot as spot_topup_conspot, execute_market
from Utility import Contract, to_builtin


def futures_execution_settlement(value, payment_fut, contract_numTask, capacity_exec,
                                 penalty_FU, cost, *, actual_demand, penalty_ESP, seed=0):
    out = execute_with_demand(value, payment_fut, contract_numTask, actual_demand,
                              capacity_exec, penalty_FU, cost, penalty_ESP,
                              seed=seed, return_details=True)
    return out["seller_utility"], float(out["buyer_utility"].sum()), out["FinishTaskNum"], out["allocation"]


def main_hybrid(value_FU_OneTask, payment_Future_min, payment_Future_max, penalty_FU_min,
                penalty_FU_max, penalty_ESP_min, penalty_ESP_max, cost_ESP, ESP_capability,
                FU_TaskNum_min, FU_TaskNum_max, value_spot, NumTask_spot,
                payment_spot_min, payment_spot_max, *, actual_demand, spot_pool, seed=0, history=None):
    nf = len(value_FU_OneTask)
    if spot_pool == "ob":
        values = np.concatenate([value_FU_OneTask, value_spot])
        demand = np.concatenate([actual_demand, NumTask_spot])
    elif spot_pool == "all":
        values, demand = np.asarray(value_spot), np.asarray(NumTask_spot)
        if not np.array_equal(values[:nf], value_FU_OneTask) or not np.array_equal(demand[:nf], actual_demand):
            raise ValueError("the all-buyer pool must start with the same FB values and gross demand")
    else:
        raise ValueError("spot_pool must be 'ob' or 'all'")
    team, ni = main_futures(value_FU_OneTask, payment_Future_min, payment_Future_max,
                           penalty_FU_min, penalty_FU_max, penalty_ESP_min, penalty_ESP_max,
                           cost_ESP, ESP_capability, FU_TaskNum_min, FU_TaskNum_max,
                           seed=seed, history=history)
    contract = Contract(team[3], *team[:3]) if team else Contract.empty(nf)
    out = execute_market(contract, demand, values, nf, ESP_capability, cost_ESP,
                         finite_grid(payment_spot_min, payment_spot_max), seed=seed)
    out.update(method="HybridFS", NI_offline=ni, NI=ni + out["NI_online"])
    return to_builtin(out)


def run(trace, config, seed=0):
    from OhTrust import evaluate_static
    return evaluate_static("HybridFS", trace, config, seed=seed)


if __name__ == "__main__":
    from OhTrust import cli_main
    cli_main(default_method="HybridFS")
