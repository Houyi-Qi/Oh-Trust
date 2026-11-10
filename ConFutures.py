import numpy as np
import BiN_CDO as CDO
from Utility import Contract, execute_futures, to_builtin
from BiN_TCD import execute_market


def generate_futures_candidates(value_FU_OneTask, payment_Future_min, payment_Future_max,
                                penalty_FU_min, penalty_FU_max, penalty_ESP_min, penalty_ESP_max,
                                cost_ESP, ESP_capability, FU_TaskNum_min, FU_TaskNum_max, **options):
    return CDO.generate_futures_candidates(value_FU_OneTask, payment_Future_min, payment_Future_max,
                                          penalty_FU_min, penalty_FU_max, penalty_ESP_min, penalty_ESP_max,
                                          cost_ESP, ESP_capability, FU_TaskNum_min, FU_TaskNum_max, **options)


def select_best_futures_candidate(candidates, cost_ESP, ESP_capability,
                                 FU_TaskNum_min=None, FU_TaskNum_max=None, OverbookRate=0.1):
    return CDO.select_best_futures_candidate(candidates, cost_ESP, ESP_capability,
                                            FU_TaskNum_min, FU_TaskNum_max, OverbookRate)


def main_futures(value_FU_OneTask, payment_Future_min, payment_Future_max,
                 penalty_FU_min, penalty_FU_max, penalty_ESP_min, penalty_ESP_max,
                 cost_ESP, ESP_capability, FU_TaskNum_min, FU_TaskNum_max, **options):
    candidates, ni = generate_futures_candidates(
        value_FU_OneTask, payment_Future_min, payment_Future_max, penalty_FU_min, penalty_FU_max,
        penalty_ESP_min, penalty_ESP_max, cost_ESP, ESP_capability, FU_TaskNum_min, FU_TaskNum_max, **options)
    best = select_best_futures_candidate(candidates, cost_ESP, ESP_capability,
                                         OverbookRate=options.get("OverbookRate", 0.1))
    if not best["feasible"]:
        return [], ni
    return [best["payment"], best["penalty_FU"], best["penalty_ESP"], best["FU_max_tasks"].tolist()], ni


def main_spot(value, payment, Contract_numTask, numTask_actual, penalty_FU, cost, *,
              capacity_exec, penalty_ESP, seed=0):
    out = execute_futures(Contract(Contract_numTask, payment, penalty_FU, penalty_ESP),
                          numTask_actual, value, capacity_exec, cost, seed=seed)
    return out["seller_utility"], float(out["buyer_utility"].sum()), out["FinishTaskNum"], 0


def main(value_FU_OneTask, payment_Future_min, payment_Future_max, penalty_FU_min, penalty_FU_max,
         penalty_ESP_min, penalty_ESP_max, cost_ESP, ESP_capability, FU_TaskNum_min, FU_TaskNum_max,
         value, numTask_actual, *, seed=0, history=None):
    if not np.array_equal(np.asarray(value), np.asarray(value_FU_OneTask)):
        raise ValueError("design and settlement valuations must refer to the same buyers")
    team, ni = main_futures(value_FU_OneTask, payment_Future_min, payment_Future_max,
                           penalty_FU_min, penalty_FU_max, penalty_ESP_min, penalty_ESP_max,
                           cost_ESP, ESP_capability, FU_TaskNum_min, FU_TaskNum_max,
                           seed=seed, history=history)
    contract = Contract(team[3], *team[:3]) if team else Contract.empty(len(value))
    out = execute_market(contract, numTask_actual, value, len(value), ESP_capability,
                         cost_ESP, (), seed=seed, spot_enabled=False)
    out.update(method="ConFutures", NI_offline=ni, NI=ni)
    return to_builtin(out)


def run(trace, config, seed=0):
    from OhTrust import evaluate_static
    return evaluate_static("ConFutures", trace, config, seed=seed)


if __name__ == "__main__":
    from OhTrust import cli_main
    cli_main(default_method="ConFutures")
