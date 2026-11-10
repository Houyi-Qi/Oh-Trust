"""Shared settlement and sample utility/risk estimates."""
from dataclasses import dataclass
import math
import numpy as np


def counts(x, name="counts", ndim=1):
    a = np.asarray(x, dtype=float)
    if a.ndim != ndim or not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a != np.floor(a)):
        raise ValueError(f"{name} must be a finite nonnegative integer array of dimension {ndim}")
    return a.astype(np.int64)


def capacity_value(capacity):
    return int(counts([capacity], "capacity")[0])


def nonnegative(x, name):
    if not np.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(x)


def values_array(values, size):
    v = np.asarray(values, dtype=float)
    if v.shape != (size,) or not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError("values must be a finite nonnegative vector matching demand")
    return v


@dataclass(frozen=True)
class Contract:
    quantities: np.ndarray
    price: float
    q_F: float
    q_E: float

    def __post_init__(self):
        n = counts(self.quantities, "contract").copy()
        n.setflags(write=False)
        object.__setattr__(self, "quantities", n)
        for name in ("price", "q_F", "q_E"):
            object.__setattr__(self, name, nonnegative(getattr(self, name), name))

    @classmethod
    def empty(cls, buyers):
        return cls(np.zeros(buyers, dtype=int), 0.0, 0.0, 0.0)


def settle_futures_scenarios(contract, demands, values, capacity, cost, priorities):
    """Settle samples with random buyer-order volunteer selection."""
    d = counts(demands, "demands", ndim=2)
    n = contract.quantities
    if d.shape[1] != len(n) or len(d) == 0:
        raise ValueError("nonempty scenarios must match the contract vector")
    v = values_array(values, len(n))
    cap, c = capacity_value(capacity), nonnegative(cost, "cost")
    priority = np.asarray(priorities, dtype=float)
    if priority.shape != d.shape or not np.all(np.isfinite(priority)):
        raise ValueError("priorities must match demand scenarios")
    x = np.minimum(n, d)
    breach = np.maximum(n - d, 0)
    excess = np.maximum(x.sum(axis=1) - cap, 0)
    order = np.argsort(priority, axis=1, kind="stable")
    ordered_x = np.take_along_axis(x, order, axis=1)
    before = ordered_x.cumsum(axis=1) - ordered_x
    ordered_vol = np.minimum(ordered_x, np.maximum(excess[:, None] - before, 0))
    volunteer = np.zeros_like(x)
    np.put_along_axis(volunteer, order, ordered_vol, axis=1)
    allocation = x - volunteer
    payment = allocation * contract.price
    fine = breach * contract.q_F
    compensation = volunteer * contract.q_E
    buyer_u = allocation * v - payment - fine + compensation
    seller_u = (payment - allocation * c + fine - compensation).sum(axis=1)
    return dict(allocation=allocation, buyer_breach=breach, volunteers=volunteer,
                buyer_utility=buyer_u, seller_utility=seller_u, payment=payment,
                buyer_fine=fine, volunteer_compensation=compensation)


def execute_futures(contract, demand, values, capacity, cost, *, seed=0, priorities=None):
    d = counts(demand, "demand")
    if priorities is None:
        priorities = np.random.default_rng(seed).random(len(d))
    batch = settle_futures_scenarios(contract, d[None, :], values, capacity, cost,
                                     np.asarray(priorities)[None, :])
    out = {k: a[0].copy() if isinstance(a[0], np.ndarray) else float(a[0])
           for k, a in batch.items()}
    out.update(demand=d.copy(), contract=contract.quantities.copy(),
               capacity=capacity_value(capacity), FinishTaskNum=int(out["allocation"].sum()))
    out["N_plus"] = out["FinishTaskNum"]
    out["N_minus"] = int((out["buyer_breach"] + out["volunteers"]).sum())
    return out


def settle_spot(allocation, residual_demand, values, price, cost, capacity):
    """Validate and settle against remaining demand/capacity, never gross demand."""
    a, d = counts(allocation, "spot allocation"), counts(residual_demand, "residual demand")
    v = values_array(values, len(d))
    cap = capacity_value(capacity)
    p, c = nonnegative(price, "price"), nonnegative(cost, "cost")
    if a.shape != d.shape or np.any(a > d) or a.sum() > cap:
        raise ValueError("spot allocation exceeds residual demand or capacity")
    if np.any((a > 0) & (v < p)) or (a.sum() and p < c):
        raise ValueError("spot price violates individual rationality")
    return dict(allocation=a, buyer_utility=a * (v - p),
                seller_utility=float(a.sum() * (p - c)), payment=a * p,
                price=p, FinishTaskNum=int(a.sum()))


def combine_market(demand, values, capacity, cost, futures, spot, num_fb, *, NI_online=0):
    d = counts(demand, "demand")
    v = values_array(values, len(d))
    if len(futures["allocation"]) != num_fb or len(spot["allocation"]) != len(d):
        raise ValueError("buyer identity/shape mismatch")
    f = np.pad(futures["allocation"], (0, len(d) - num_fb))
    s = spot["allocation"]
    a = f + s
    if np.any(a > d) or a.sum() > capacity or np.any(a < 0):
        raise ValueError("duplicate tasks or capacity violation")
    buyer_u = np.pad(futures["buyer_utility"], (0, len(d) - num_fb)) + spot["buyer_utility"]
    seller_u = float(futures["seller_utility"] + spot["seller_utility"])
    surplus = float(a @ (v - cost))
    if not np.isclose(buyer_u.sum() + seller_u, surplus, atol=1e-8):
        raise AssertionError("money transfers do not balance")
    return dict(demand=d.copy(), allocation=a, futures_allocation=f, spot_allocation=s,
                buyer_breach=futures["buyer_breach"], volunteers=futures["volunteers"],
                buyer_fine=futures["buyer_fine"],
                volunteer_compensation=futures["volunteer_compensation"],
                buyer_utility=buyer_u, ESP_Utility=seller_u, MU_Utility=float(buyer_u.sum()),
                futures_ESP_Utility=float(futures["seller_utility"]),
                futures_MU_Utility=float(futures["buyer_utility"].sum()),
                FinishTaskNum=int(a.sum()), FinishTaskNum_futures=int(f.sum()),
                FinishTaskNum_spot=int(s.sum()), N_plus=futures["N_plus"],
                N_minus=futures["N_minus"], NI_online=int(NI_online), NI_offline=0,
                NI=int(NI_online), capacity=int(capacity), surplus=surplus)


def to_builtin(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [to_builtin(v) for v in obj]
    return obj


def summarize_records(records, NI_offline, method, trajectory_hash):
    keys = ("ESP_Utility", "MU_Utility", "FinishTaskNum", "FinishTaskNum_futures",
            "FinishTaskNum_spot", "NI_online", "N_plus", "N_minus")
    out = {key: sum(r[key] for r in records) for key in keys}
    out.update(method=method, trajectory_hash=trajectory_hash, transactions=len(records),
               NI_offline=int(NI_offline), NI=int(NI_offline + out["NI_online"]),
               BuyerBreachTasks=sum(int(r["buyer_breach"].sum()) for r in records),
               VolunteerTasks=sum(int(r["volunteers"].sum()) for r in records),
               BuyerFines=sum(float(r["buyer_fine"].sum()) for r in records),
               VolunteerCompensation=sum(float(r["volunteer_compensation"].sum()) for r in records))
    return to_builtin(out)


@dataclass(frozen=True)
class RiskLimits:
    rho1: float = 0.3
    rho2: float = 0.3
    rho3: float = 0.3
    buyer_utility_min: float = 1e-9
    seller_target: float = 0.0  # DEMO only; paper uses 2400.

    def __post_init__(self):
        if any(not np.isfinite(x) or x < 0 or x > 1 for x in (self.rho1, self.rho2, self.rho3)):
            raise ValueError("risk limits must be in [0,1]")
        if not np.isfinite(self.buyer_utility_min) or not np.isfinite(self.seller_target):
            raise ValueError("utility thresholds must be finite")


def evaluate_contract(contract, scenarios, values, capacity, cost, priorities, limits=None):
    from Pr import capacity_probabilities
    limits = limits or RiskLimits()
    out = settle_futures_scenarios(contract, scenarios, values, capacity, cost, priorities)
    return dict(expected_buyer_utility=out["buyer_utility"].mean(axis=0),
                expected_seller_utility=float(out["seller_utility"].mean()),
                risk_buyer_utility=(out["buyer_utility"] < limits.buyer_utility_min).mean(axis=0),
                risk_volunteer=(out["volunteers"] > 0).mean(axis=0),
                risk_seller=float(np.mean(out["seller_utility"] < limits.seller_target)),
                **capacity_probabilities(contract.quantities, scenarios, capacity))


def U_ESP(num_futures, num_spot, payment_futures, payment_spot,
          q_ESP, q_FU, cost, num_volunteer, num_FU_break):
    return (num_futures * (payment_futures - cost) + num_spot * (payment_spot - cost)
            - q_ESP * num_volunteer + q_FU * num_FU_break)


def U_F_i(num_task_futures, num_task_spot, num_volunteer, num_task_break,
          payment_futures, payment_spot, value, q_ESP, q_FU):
    return (num_task_futures * (value - payment_futures)
            + num_task_spot * (value - payment_spot)
            + num_volunteer * q_ESP - num_task_break * q_FU)


def Utility_OU(num, payment_spot, value):
    return num * (value - payment_spot)


def binomial_probability(n, k, p):
    return 0.0 if k < 0 or k > n else math.comb(n, k) * p**k * (1 - p)**(n - k)


def U_F_i_expected(*args, **kwargs):
    raise NotImplementedError("Unsafe legacy expectation removed; use evaluate_contract with joint scenarios")


def U_ESP_Expected(*args, **kwargs):
    raise NotImplementedError("TotalContract cannot identify a distribution; use evaluate_contract")
