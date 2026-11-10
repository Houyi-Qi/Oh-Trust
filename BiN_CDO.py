"""Finite contract offers with joint sample-risk screening."""
from dataclasses import dataclass, field
from itertools import product
import math
import numpy as np
import Pr
from Utility import RiskLimits, evaluate_contract
from Utility import Contract, counts, capacity_value, nonnegative, values_array, execute_futures


def finite_grid(lo, hi, step=1.0):
    lo, hi = nonnegative(lo, "lower bound"), nonnegative(hi, "upper bound")
    if hi < lo or not np.isfinite(step) or step <= 0:
        raise ValueError("invalid grid bounds/step")
    grid = list(np.arange(lo, hi + 1e-10, step))
    if not grid or grid[-1] < hi - 1e-9:
        grid.append(hi)
    return tuple(sorted(set(float(round(x, 10)) for x in grid), reverse=True))


@dataclass(frozen=True)
class DesignConfig:
    prices: tuple = (3.0, 4.0, 5.0, 6.0)
    buyer_fines: tuple = (1.0, 2.0)
    seller_compensations: tuple = (1.0, 2.0)
    overbook_rate: float = 0.1
    risks: RiskLimits = field(default_factory=RiskLimits)
    samples: int = 64
    max_joint_vectors: int = 20000
    search_mode: str = "auto"
    max_capacity_states: int = 10000

    def __post_init__(self):
        nonnegative(self.overbook_rate, "overbook_rate")
        for name in ("prices", "buyer_fines", "seller_compensations"):
            grid = tuple(sorted(set(nonnegative(x, name) for x in getattr(self, name)), reverse=True))
            if not grid:
                raise ValueError(f"{name} grid is empty")
            object.__setattr__(self, name, grid)
        if any(x < 1 or int(x) != x for x in (self.samples, self.max_joint_vectors, self.max_capacity_states)):
            raise ValueError("samples and max_joint_vectors must be positive integers")
        if self.search_mode not in ("auto", "exhaustive", "bounded"):
            raise ValueError("search_mode must be auto, exhaustive or bounded")


def risk_FU(risk_probability, threshold=0.3):
    """Chance constraint, not a comparison between mean utility and probability."""
    return 0 <= risk_probability <= threshold


def _bounded_vectors(values, scenarios, options, price, fine, cost, capacity, limit, risks):
    score = np.full(limit + 1, -np.inf)
    score[0] = 0.0
    choices = np.zeros((len(values), limit + 1), dtype=int)
    ni = 0
    for i, quantities in enumerate(options):
        updated = score.copy()
        for n in quantities:
            if n == 0:
                continue
            ni += 2
            if values[i] < price:
                continue
            x = np.minimum(n, scenarios[:, i])
            breach = np.maximum(n - scenarios[:, i], 0)
            buyer_u = x * (values[i] - price) - breach * fine
            if buyer_u.mean() < 0 or np.mean(buyer_u < risks.buyer_utility_min) > risks.rho1:
                continue
            # Additive proposals omit overload; the full joint ledger screens them below.
            gain = float(np.mean(x * (price - cost) + breach * fine))
            proposal = score[:-n] + gain
            improve = proposal > updated[n:]
            updated[n:][improve] = proposal[improve]
            choices[i, n:][improve] = n
        score = updated
    budgets = sorted({min(limit, capacity), (min(limit, capacity) + limit) // 2, limit})
    vectors, seen = [], set()
    for budget in budgets:
        weight = int(np.argmax(score[:budget + 1]))
        n = np.zeros(len(values), dtype=int)
        for i in range(len(values) - 1, -1, -1):
            n[i] = choices[i, weight]
            weight -= int(n[i])
        key = tuple(n)
        if n.sum() and key not in seen:
            vectors.append(n)
            seen.add(key)
    return vectors, ni


def search_candidates(values, capacity, cost, lower, upper, *, history=None,
                      scenarios=None, config=None, seed=0):
    cfg = config or DesignConfig()
    cap, cost = capacity_value(capacity), nonnegative(cost, "cost")
    lo, hi = counts(lower, "lower contract bounds"), counts(upper, "upper contract bounds")
    v = values_array(values, len(lo))
    if lo.shape != hi.shape or np.any(lo > hi):
        raise ValueError("invalid contract bounds")
    if scenarios is None:
        scenarios = Pr.sample_demand_scenarios(history=history, lower=lo, upper=hi,
                                               samples=cfg.samples, seed=seed)
    d = counts(scenarios, "scenarios", ndim=2)
    if not len(d) or d.shape[1] != len(v):
        raise ValueError("scenario dimensions do not match buyers")
    priorities = np.random.default_rng(np.random.SeedSequence([seed, 991])).random(d.shape)
    sample_id = Pr.scenario_digest(d, priorities)
    limit = math.floor((1 + cfg.overbook_rate) * cap + 1e-9)
    options = [list(range(min(int(b), limit), max(1, int(a)) - 1, -1)) + [0] for a, b in zip(lo, hi)]
    count = math.prod(len(o) for o in options)
    bounded = cfg.search_mode == "bounded" or (cfg.search_mode == "auto" and count > cfg.max_joint_vectors)
    if count > cfg.max_joint_vectors and not bounded:
        raise ValueError(f"joint grid has {count} vectors; limit {cfg.max_joint_vectors}. "
                         "Use search_mode='auto' or 'bounded'.")
    if bounded and limit + 1 > cfg.max_capacity_states:
        raise ValueError("capacity exceeds max_capacity_states for bounded search")
    exact_vectors = [] if bounded else [np.asarray(n, dtype=int) for n in product(*options) if 0 < sum(n) <= limit]
    candidates, interactions = [], 0
    for p, qf, qe in product(cfg.prices, cfg.buyer_fines, cfg.seller_compensations):
        if p < cost:
            continue
        if bounded:
            vectors, ni = _bounded_vectors(v, d, options, p, qf, cost, cap, limit, cfg.risks)
            interactions += ni
        else:
            vectors = exact_vectors
        for n in vectors:
            active = n > 0
            interactions += int(active.sum()) + 1  # proposal/reply proxy, not measured packets
            if np.any(v[active] < p):
                continue
            contract = Contract(n, p, qf, qe)
            risk = evaluate_contract(contract, d, v, cap, cost, priorities, cfg.risks)
            buyer_ok = (np.all(risk["risk_buyer_utility"][active] <= cfg.risks.rho1)
                        and np.all(risk["risk_volunteer"][active] <= cfg.risks.rho2)
                        and np.all(risk["expected_buyer_utility"][active] >= 0))
            if buyer_ok:
                candidates.append(dict(payment=p, penalty_FU=qf, penalty_ESP=qe,
                                       FU_max_tasks=n, contract=contract, risk=risk,
                                       ESP_EUtility=risk["expected_seller_utility"],
                                       seller_risk_limit=cfg.risks.rho3, sample_id=sample_id,
                                       evaluation_capacity=cap, evaluation_cost=cost,
                                       search_mode="bounded" if bounded else "exhaustive"))
    return candidates, interactions


def select_best_futures_candidate(candidates, cost_ESP, ESP_capability,
                                 K_min=None, K_max=None, OverbookRate=0.1):
    cap = capacity_value(ESP_capability)
    limit = math.floor((1 + nonnegative(OverbookRate, "OverbookRate")) * cap + 1e-9)
    feasible = []
    for cand in candidates:
        if cand.get("evaluation_capacity") != cap or cand.get("evaluation_cost") != cost_ESP:
            raise ValueError("candidate evaluated with different resources; rebuild the scenarios")
        if (cand["FU_max_tasks"].sum() <= limit
                and cand["risk"]["risk_seller"] <= cand["seller_risk_limit"]
                and cand["ESP_EUtility"] >= 0):
            feasible.append(cand)
    if not feasible:
        return {"feasible": False, "reason": "No grid candidate satisfies all sample risk constraints"}
    # First candidate breaks exact ties; grids and quantity order are deterministic.
    return {**max(feasible, key=lambda c: c["ESP_EUtility"]), "feasible": True}


def design_contract(values, history, capacity, cost, *, config=None, seed=0):
    cfg = config or DesignConfig()
    h = counts(history, "history", ndim=2)
    if not len(h):
        raise ValueError("contract design requires pre-transaction history")
    candidates, ni = search_candidates(values, capacity, cost, h.min(axis=0), h.max(axis=0),
                                      history=h, config=cfg, seed=seed)
    best = select_best_futures_candidate(candidates, cost, capacity, OverbookRate=cfg.overbook_rate)
    best["NI_offline"] = ni
    best["history_rows"] = len(h)
    best["history_bounds"] = [h.min(axis=0).tolist(), h.max(axis=0).tolist()]
    if not best["feasible"]:
        best["contract"] = Contract.empty(h.shape[1])
    return best


def generate_futures_candidates(value_FU_OneTask, payment_min, payment_max,
                                penalty_FU_min, penalty_FU_max, penalty_ESP_min, penalty_ESP_max,
                                cost_ESP, ESP_capability, K_min, K_max, *,
                                history=None, scenarios=None, seed=0, risks=None,
                                OverbookRate=0.1, samples=64, search_mode="auto"):
    size = len(value_FU_OneTask)
    lo = np.broadcast_to(K_min, (size,))
    hi = np.broadcast_to(K_max, (size,))
    cfg = DesignConfig(finite_grid(payment_min, payment_max),
                       finite_grid(penalty_FU_min, penalty_FU_max),
                       finite_grid(penalty_ESP_min, penalty_ESP_max),
                       OverbookRate, risks or RiskLimits(), samples, search_mode=search_mode)
    return search_candidates(value_FU_OneTask, ESP_capability, cost_ESP, lo, hi,
                             history=history, scenarios=scenarios, config=cfg, seed=seed)


def execute_with_demand(value, payment_fut, contract_numTask, actual_demand,
                        capacity_exec, penalty_FU, cost, penalty_ESP=0.0, *,
                        seed=0, return_details=False):
    out = execute_futures(Contract(contract_numTask, payment_fut, penalty_FU, penalty_ESP),
                          actual_demand, value, capacity_exec, cost, seed=seed)
    if return_details:
        return out
    return (out["seller_utility"], float(out["buyer_utility"].sum()),
            out["FinishTaskNum"], out["allocation"], out["N_plus"], out["N_minus"])
