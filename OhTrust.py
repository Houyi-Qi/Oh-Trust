"""Causal Execute/Renew environment and held-out-validation DDQN training."""
from dataclasses import dataclass, field, asdict, replace
import argparse
import json
from pathlib import Path
import copy
import numpy as np
import BiN_CDO as CDO
import BiN_TCD as TCD
import SRU_ConR as SRU
from data_utils import synthetic_trace, load_trace_csv, DemandTrace
from Utility import (Contract, counts, values_array, execute_futures, combine_market,
                     capacity_value, nonnegative, summarize_records, to_builtin)
from Random import random_spot


@dataclass(frozen=True)
class MarketConfig:
    capacity: int = 10                 # SYNTHETIC stress setting, not paper G=600
    cost: float = 2.0
    design: CDO.DesignConfig = field(default_factory=CDO.DesignConfig)
    spot_prices: tuple = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    history_window: int = 12
    max_invalid_renewals: int = 2
    w4: float = 1.0
    w5: float = 1.0
    omega6: float = 1e-3
    omega7: float = 1.0
    renew_reward: float = -10.0
    spot_limits: tuple = ()

    def __post_init__(self):
        capacity_value(self.capacity)
        nonnegative(self.cost, "cost")
        if self.history_window < 1 or self.max_invalid_renewals < 1:
            raise ValueError("positive history/renewal limits required")
        for p in self.spot_prices:
            nonnegative(p, "spot price")
        counts(self.spot_limits, "spot_limits")
        if self.renew_reward >= 0:
            raise ValueError("Eq.22 requires a negative renewal reward")


class OhTrustEnv:
    """Four observed state values; only Execute advances the demand cursor."""
    state_size, action_size = 4, 2

    def __init__(self, trace=None, config=None, seed=0):
        self.trace = trace if trace is not None else synthetic_trace(seed)
        self.config = config or MarketConfig()
        self.seed = seed
        self._initialized = False

    def set_settings_tuple(self, settings, *, history=None, demands=None, horizon=100):
        if self._initialized and not self.done:
            raise RuntimeError("configure inputs before reset()")
        (self.Num_FU, _, _, _, self.FU_TaskNum_min, self.FU_TaskNum_max, _,
         self.cost_ESP, self.ESP_capability, _, _, self.value_FU_OneTask,
         self.payment_Future_min, self.payment_Future_max, self.penalty_FU_min,
         self.penalty_FU_max, self.penalty_ESP_min, self.penalty_ESP_max) = settings
        rng = np.random.default_rng(self.seed)
        shape = (self.config.history_window, self.Num_FU)
        if history is None:
            history = rng.integers(self.FU_TaskNum_min, self.FU_TaskNum_max + 1, size=shape)
        if demands is None:
            demands = rng.integers(self.FU_TaskNum_min, self.FU_TaskNum_max + 1, size=(horizon, self.Num_FU))
        self.trace = DemandTrace(history, demands, self.value_FU_OneTask, self.Num_FU,
                                 "settings tuple; synthetic defaults unless observations supplied")
        design = replace(self.config.design,
                         prices=CDO.finite_grid(self.payment_Future_min, self.payment_Future_max),
                         buyer_fines=CDO.finite_grid(self.penalty_FU_min, self.penalty_FU_max),
                         seller_compensations=CDO.finite_grid(self.penalty_ESP_min, self.penalty_ESP_max))
        self.config = replace(self.config, capacity=self.ESP_capability, cost=self.cost_ESP,
                              design=design, spot_prices=design.prices, spot_limits=())
        self._initialized = False

    def set_spot_inputs(self, value_spot, NumTask_spot, payment_spot_min, payment_spot_max):
        if self._initialized and not self.done:
            raise RuntimeError("configure inputs before reset()")
        if not np.array_equal(np.asarray(value_spot), self.trace.values):
            raise ValueError("spot buyers must match trace identities; put OB demand in DemandTrace")
        limits = counts(NumTask_spot, "spot request limits")
        if len(limits) != len(self.trace.values):
            raise ValueError("spot limits must match trace buyers")
        self.config = replace(self.config, spot_prices=CDO.finite_grid(payment_spot_min, payment_spot_max),
                              spot_limits=tuple(limits.tolist()))

    def _pick_contract(self):
        best = CDO.design_contract(self.trace.values[:self.trace.num_fb],
                                   self.history[:, :self.trace.num_fb], self.config.capacity,
                                   self.config.cost, config=self.config.design, seed=self.seed)
        self.NI_offline += best["NI_offline"]
        self.current_contract = best["contract"]
        self.contract_design = best
        self.contract_version = self.history_version
        return best["NI_offline"]

    def reset(self):
        self.history = self.trace.history[-self.config.history_window:].copy()
        self.t = self.history_version = self.contract_version = 0
        self.NI_offline = self.NI_online = self.renewals = self.invalid_renewals = 0
        self.records, self.events = [], []
        self.episode_return = 0.0
        self.done, self.truncated = False, False
        ni = self._pick_contract()
        self._state = np.array([self.history[:, :self.trace.num_fb].sum(axis=1).mean(),
                                0.0, ni, 0.0], dtype=np.float32)
        self._initialized = True
        return self._state.copy()

    def valid_actions(self):
        if self.done:
            return ()
        return (0, 1) if self.history_version > self.contract_version else (1,)

    def state_scale(self):
        # Public resource constants and warmup history only; no test-demand fit.
        cap = max(1, self.config.capacity)
        value = max(1.0, float(np.max(self.trace.values, initial=0)))
        return np.array([cap, cap * value, max(1, self.contract_design["NI_offline"]),
                         max(1, cap * (1 + self.config.design.overbook_rate))], dtype=float)

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("call reset() first")
        if self.done:
            raise RuntimeError("episode already finished")
        if action not in (0, 1):
            raise ValueError("action must be Renew=0 or Execute=1")
        info = dict(action=int(action), executed=False, transaction_index=self.t,
                    FinishTaskNum=0, FinishTaskNum_futures=0, FinishTaskNum_spot=0,
                    NI_offline=0, NI_online=0, NI=0, ESP_Utility=0.0, MU_Utility=0.0)
        esp = mu = 0.0
        if action == 0:
            if 0 not in self.valid_actions():
                self.invalid_renewals += 1
                info["renewal_blocked"] = "no new history since previous contract search"
                if self.invalid_renewals >= self.config.max_invalid_renewals:
                    self.done = self.truncated = True
            else:
                ni = self._pick_contract()
                self.renewals += 1
                self.invalid_renewals = 0
                self._state[2] = ni
                info.update(NI_offline=ni, NI=ni, contract_changed_at_history=self.history_version)
            r = SRU.reward(0, 0, 0, 0, Rew1=self.config.renew_reward)
        else:
            # Sole future-trajectory read in the entire environment decision path.
            demand = self.trace.demands[self.t]
            trade_seed = np.random.SeedSequence([self.seed, self.t, 17])
            trade = TCD.execute_market(self.current_contract, demand, self.trace.values,
                                       self.trace.num_fb, self.config.capacity, self.config.cost,
                                       self.config.spot_prices, seed=trade_seed,
                                       spot_limits=self.config.spot_limits)
            self.records.append(trade)
            self.NI_online += trade["NI_online"]
            rep = SRU.reputation(trade["N_plus"], trade["N_minus"], self.config.w4, self.config.w5)
            esp, mu = trade["ESP_Utility"], trade["MU_Utility"]
            # Eq.22 uses futures-side U_dot + U_tilde; spot utilities are reported separately.
            r = SRU.reward(1, trade["futures_ESP_Utility"], trade["futures_MU_Utility"], rep,
                           omega6=self.config.omega6, omega7=self.config.omega7)
            self._state = np.array([np.sum(demand[:self.trace.num_fb]), esp + mu,
                                    trade["NI_online"], rep], dtype=np.float32)
            self.history = np.concatenate([self.history, np.asarray(demand)[None, :]])[-self.config.history_window:]
            self.t += 1
            self.history_version += 1
            self.invalid_renewals = 0
            self.done = self.t == len(self.trace.demands)
            info.update(trade, executed=True, reputation=rep)
        self.episode_return += r
        info.update(history_version=self.history_version, contract_version=self.contract_version,
                    contract=self.current_contract.quantities.copy(),
                    price=self.current_contract.price, q_F=self.current_contract.q_F,
                    q_E=self.current_contract.q_E, contract_feasible=self.contract_design["feasible"],
                    terminated=self.done and not self.truncated, truncated=self.truncated,
                    valid_actions=self.valid_actions(), NI_offline_total=self.NI_offline,
                    NI_online_total=self.NI_online)
        self.events.append(to_builtin(info))
        return self._state.copy(), float(r), self.done, info, float(esp), float(mu)

    def summary(self):
        out = summarize_records(self.records, self.NI_offline, "OhTrust", self.trace.digest)
        out.update(renewals=self.renewals, episode_return=float(self.episode_return),
                   terminated=self.done and not self.truncated, truncated=self.truncated)
        return out


def run_episode(env, agent, *, training=False):
    state = env.reset()
    agent.set_training(training)
    losses = []
    while not env.done:
        a = agent.act(state, evaluate=not training, valid_actions=env.valid_actions())
        next_state, r, done, info, _, _ = env.step(a)
        if training:
            agent.remember(state, a, r, next_state, done, env.valid_actions())
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)
        state = next_state
    result = env.summary()
    result["training_updates"] = len(losses)
    return result


def train_agent(train_trace, validation_trace, config=None, *, episodes=3, seed=0,
                best_path=None, batch_size=8):
    """Select weights on validation return; no test trajectory is accepted here."""
    if episodes < 1:
        raise ValueError("at least one training episode is required")
    if train_trace.digest == validation_trace.digest:
        raise ValueError("training and validation trajectories must be independent")
    if (train_trace.num_fb != validation_trace.num_fb
            or not np.array_equal(train_trace.values, validation_trace.values)):
        raise ValueError("training and validation must have the same buyer/resource schema")
    cfg = config or MarketConfig()
    train_env = OhTrustEnv(train_trace, cfg, seed)
    train_env.reset()
    agent = SRU.DDQNAgent(seed=seed, state_scale=train_env.state_scale(), batch_size=batch_size)
    best_score, best_weights, best_episode = -np.inf, None, None
    train_results, validation_results = [], []
    for episode in range(episodes):
        train_results.append(run_episode(train_env, agent, training=True))
        validation = run_episode(OhTrustEnv(validation_trace, cfg, seed), agent, training=False)
        validation_results.append(validation)
        score = validation["episode_return"]
        if score > best_score:
            best_score, best_episode = score, episode + 1
            best_weights = copy.deepcopy(agent.model.state_dict())
    agent.model.load_state_dict(best_weights)
    agent.update_target(tau=1.0)
    agent.set_training(False)
    if best_path is not None:
        agent.save(best_path)
    return agent, dict(selected_episode=best_episode, selection_metric="validation episode return",
                       training=train_results, validation=validation_results,
                       train_trajectory_hash=train_trace.digest, validation_trajectory_hash=validation_trace.digest,
                       optimizer_updates=agent.updates, evaluation_exploration=False)


def train_and_save(episodes=3, steps_per_episode=12, best_path="ohtrust_ddqn_best.pt", *, seed=0):
    traces = [synthetic_trace(seed + offset, slots=steps_per_episode) for offset in (0, 10000, 20000)]
    agent, report = train_agent(traces[0], traces[1], episodes=episodes, seed=seed, best_path=best_path)
    test = run_episode(OhTrustEnv(traces[2], seed=seed), agent)
    return {"training_report": report, "held_out_test": test}



def demo_settings(Num_FU=30, *, seed=0):
    rng = np.random.default_rng(seed)
    local_rate = rng.integers(10, 15, Num_FU) * 1e8 / 600
    transmit_power = rng.integers(500, 551, Num_FU) * 0.001
    local_power = rng.integers(450, 501, Num_FU) * 0.001
    server_rate, task_size = 1.5e12 / 600, 1.5e6
    local_time = task_size / local_rate
    transmit_time = task_size / (6e6 * np.log2(1 + transmit_power * 250))
    time_saved = local_time - transmit_time - task_size / server_rate
    energy_saved = local_time * local_power - transmit_time * transmit_power
    values = 10 * time_saved + 10 * energy_saved
    cost = 10 * task_size * 0.7 / server_rate + 0.001
    return (Num_FU, local_rate, local_power, 20, 7, 12, transmit_power, cost, 600,
            10., 10., values, float(values.min()), float(values.max()), 0.5, 3., 0.5, 5.)


METHODS = ("OhTrust", "ConFutures", "HybridFS", "ConSpot", "Random")


def evaluate_static(method, trace, config, *, seed=0):
    if method not in METHODS[1:]:
        raise ValueError("unknown static method")
    offline = 0
    contract = Contract.empty(trace.num_fb)
    design = None
    if method in ("ConFutures", "HybridFS"):
        design = CDO.design_contract(trace.values[:trace.num_fb],
                                 trace.history[-config.history_window:, :trace.num_fb],
                                 config.capacity, config.cost, config=config.design, seed=seed)
        contract, offline = design["contract"], design["NI_offline"]
    records = []
    for t, demand in enumerate(trace.demands):
        trial_seed = np.random.SeedSequence([seed, t, 17])
        if method in ("ConFutures", "HybridFS"):
            out = TCD.execute_market(contract, demand, trace.values, trace.num_fb, config.capacity,
                                 config.cost, config.spot_prices, seed=trial_seed,
                                 spot_enabled=(method == "HybridFS"), spot_limits=config.spot_limits)
        else:
            futures = execute_futures(contract, demand[:trace.num_fb], trace.values[:trace.num_fb],
                                       config.capacity, config.cost, seed=trial_seed)
            clearing = TCD.clear_spot if method == "ConSpot" else random_spot
            kwargs = {"seed": np.random.SeedSequence([seed, t, 31])} if method == "Random" else {}
            requests = np.minimum(demand, config.spot_limits) if config.spot_limits else demand
            spot = clearing(trace.values, requests, config.cost, config.capacity, config.spot_prices, **kwargs)
            out = combine_market(demand, trace.values, config.capacity, config.cost, futures, spot,
                                 trace.num_fb, NI_online=spot["NI_online"])
        records.append(out)
    result = summarize_records(records, offline, method, trace.digest)
    result.update(renewals=0, terminated=True, truncated=False)
    if design is not None:
        result["initial_contract"] = dict(quantities=contract.quantities.tolist(), price=contract.price,
                                          q_F=contract.q_F, q_E=contract.q_E,
                                          feasible=design["feasible"],
                                          history_bounds=design["history_bounds"])
    return result, records


def compare(traces, config=None, *, seed=7, episodes=3, checkpoint=None, methods=METHODS):
    cfg = config or MarketConfig()
    if len({traces[name].digest for name in ("train", "validation", "test")}) != 3:
        raise ValueError("train, validation and test must be distinct")
    summaries, raw_records, policy_report = {}, {}, None
    for method in methods:
        if method == "OhTrust":
            agent, policy_report = train_agent(traces["train"], traces["validation"], cfg,
                                               episodes=episodes, seed=seed, best_path=checkpoint)
            env = OhTrustEnv(traces["test"], cfg, seed)
            result = run_episode(env, agent, training=False)
            result["policy"] = "DDQN, validation-selected, epsilon=0 evaluation"
            records = env.records
            events = env.events
        else:
            result, records = evaluate_static(method, traces["test"], cfg, seed=seed)
        summaries[method], raw_records[method] = result, records
    shared = all(len(rows) == len(traces["test"].demands)
                 and all(np.array_equal(row["demand"], traces["test"].demands[t])
                         for t, row in enumerate(rows)) for rows in raw_records.values())
    if not shared:
        raise AssertionError("methods did not consume the identical demand trajectory")
    return to_builtin(dict(scope="MINIMAL_SYNTHETIC_DEMO" if traces["test"].label.startswith("synthetic")
                           else "LOCAL_DEMANDS_WITH_DEMO_VALUATIONS",
                           evaluation_scope="demonstration_only",
                           seed=seed, config=asdict(cfg), shared_trajectory_verified=shared,
                           traces={name: {"hash": tr.digest, "source": tr.label,
                                           "transactions": len(tr.demands),
                                           "warmup_rows": len(tr.history)} for name, tr in traces.items()},
                           summaries=summaries, training=policy_report, records=raw_records,
                           ohtrust_events=events if "OhTrust" in methods else [],
                           validation_checks={"all_methods_share_test_trace": "PASS"}))


def cli_main(default_method="OhTrust"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--model", default="ohtrust_ddqn_best.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=10, help="demo capacity, paper uses 600")
    parser.add_argument("--output", default="results/compare.json")
    parser.add_argument("--csv", help="local chronological Date + buyer demand columns")
    parser.add_argument("--raw-taxi", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--history-window", type=int, help="past records used by each contract search")
    parser.add_argument("--num-fb", type=int, default=3)
    args = parser.parse_args()
    if args.csv:
        traces = load_trace_csv(args.csv, num_fb=args.num_fb, warmup=args.warmup, raw_taxi=args.raw_taxi)
    else:
        traces = {name: synthetic_trace(args.seed + offset, slots=args.slots, num_fb=args.num_fb)
                  for name, offset in zip(("train", "validation", "test"), (0, 10000, 20000))}
    window = args.history_window if args.history_window is not None else (args.warmup if args.csv else 12)
    cfg = MarketConfig(capacity=args.capacity, history_window=window)
    cfg = replace(cfg, design=replace(cfg.design, samples=args.samples))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = METHODS if args.compare else (default_method,)
    result = compare(traces, cfg, seed=args.seed, episodes=args.episodes,
                     checkpoint=args.model, methods=methods)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summaries"], indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Comparison completed. Results saved to {path}.")



if __name__ == "__main__":
    cli_main()
