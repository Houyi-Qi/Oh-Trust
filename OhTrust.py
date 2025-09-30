# Oh-Trust.py
# ----------------------------------------------------
# Oh-Trust main runner:
# - State: 4D [demand, total_utility, NI, reputation]
# - Actions: 0 = Renew (pick new futures contract)
#            1 = Execute (futures execution + spot top-up)
# - Reward: SRU_ConR.reward() (consistent with the paper)
# - Training: DDQN; best model saved as ohtrust_ddqn_best.pt
# ----------------------------------------------------

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import numpy as np

import BiN_CDO as CDO           # Futures trading (candidate search, execution w/ demand)
import BiN_TCD as TCD           # Spot trading (ConSpot)
import SRU_ConR as SRU          # Reputation, reward, and DDQN agent


class OhTrustEnv:
    """
    Oh-Trust Environment
    --------------------
    A lightweight training environment that integrates:
      - Futures design/selection via BiN_CDO
      - Execution with realized demand & capacity limits
      - Spot-stage top-up via BiN_TCD when capacity remains
      - Reputation and reward calculation via SRU_ConR

    Observation/State vector (float32):
        [ demand_total, total_utility_sum, NI, reputation ]

    Action space:
        0 -> Renew: re-pick the best futures contract (no execution)
        1 -> Execute: run futures execution + spot top-up this step

    Notes:
        * This is a minimal training shell for reproducibility, not a full gym env.
        * External APIs expected:
            - CDO.generate_futures_candidates(...)
            - CDO.select_best_futures_candidate(...)
            - CDO.execute_with_demand(...)
            - TCD.conspot(...)
            - SRU.reputation(...), SRU.reward(...)
            - SRU.DDQNAgent(...)
    """

    def __init__(self):
        # -------- Futures settings (will be populated via set_settings_tuple) --------
        self.Num_FU: Optional[int] = None
        self.FU_TaskNum_min: Optional[int] = None
        self.FU_TaskNum_max: Optional[int] = None
        self.cost_ESP: Optional[float] = None
        self.ESP_capability: Optional[int] = None
        self.value_FU_OneTask: Optional[np.ndarray] = None
        self.payment_Future_min: Optional[float] = None
        self.payment_Future_max: Optional[float] = None
        self.penalty_FU_min: Optional[float] = None
        self.penalty_FU_max: Optional[float] = None
        self.penalty_ESP_min: Optional[float] = None
        self.penalty_ESP_max: Optional[float] = None

        # -------- Spot settings (optional; defaults to futures values if not set) --------
        self.value_spot: Optional[np.ndarray] = None
        self.NumTask_spot: Optional[np.ndarray] = None
        self.payment_spot_min: Optional[float] = None
        self.payment_spot_max: Optional[float] = None

        # -------- RL config --------
        self.state_size = 4
        self.action_size = 2
        self.max_steps = 100  # not enforced here; controlled by the trainer

        # -------- Reputation/Reward hyper-parameters (consistent with paper) --------
        self.Rew1 = -10.0          # base penalty when action = Renew
        self.w4, self.w5 = 1.0, 1.0
        self.omega6, self.omega7 = 1e-3, 1.0

        # -------- Internal trackers --------
        self.current_contract: Optional[Dict[str, Any]] = None
        self._last_total_utility = 0.0
        self._last_NI = 0
        self._last_reputation = 0.0
        self._last_demand = 0

    # ---------------------------
    # Configuration helpers
    # ---------------------------
    def set_settings_tuple(self, settings: Tuple):
        """
        Populate futures-related settings from a pre-packed tuple.
        See demo_settings(...) for the tuple layout.
        """
        (self.Num_FU, _FU_comp, _FU_pow_local, _Num_OU_expected,
         self.FU_TaskNum_min, self.FU_TaskNum_max, _FU_power_trans,
         self.cost_ESP, self.ESP_capability, _V1, _V2, self.value_FU_OneTask,
         self.payment_Future_min, self.payment_Future_max,
         self.penalty_FU_min, self.penalty_FU_max,
         self.penalty_ESP_min, self.penalty_ESP_max) = settings

    def set_spot_inputs(self,
                        value_spot,
                        NumTask_spot,
                        payment_spot_min: float,
                        payment_spot_max: float) -> None:
        """
        Set spot-stage pool and price scan bounds.

        Args:
            value_spot: array-like of per-task valuations for spot candidates
            NumTask_spot: array-like of max tasks per candidate
            payment_spot_min/max: scan bounds for spot-stage uniform price
        """
        self.value_spot = np.asarray(value_spot, dtype=float)
        self.NumTask_spot = np.asarray(NumTask_spot, dtype=int)
        self.payment_spot_min = float(payment_spot_min)
        self.payment_spot_max = float(payment_spot_max)

    # ---------------------------
    # Internal utilities
    # ---------------------------
    def _ensure(self) -> None:
        """Validate that required settings are present; set spot defaults if absent."""
        req = [self.Num_FU, self.FU_TaskNum_min, self.FU_TaskNum_max, self.cost_ESP,
               self.ESP_capability, self.value_FU_OneTask, self.payment_Future_min,
               self.payment_Future_max, self.penalty_FU_min, self.penalty_FU_max,
               self.penalty_ESP_min, self.penalty_ESP_max]
        if any(x is None for x in req):
            raise RuntimeError("Settings incomplete. Call set_settings_tuple(...) first.")

        if self.value_spot is None:
            # Default: reuse futures valuations; one task per candidate
            self.set_spot_inputs(
                value_spot=self.value_FU_OneTask,
                NumTask_spot=np.ones(int(self.Num_FU), dtype=int),
                payment_spot_min=float(self.payment_Future_min),
                payment_spot_max=float(self.payment_Future_max),
            )

    def _pick_contract(self) -> Optional[Dict[str, Any]]:
        """
        Run futures candidate search + selection once and cache NI.
        Returns:
            Best feasible contract dict or None.
        """
        cand, NI = CDO.generate_futures_candidates(
            self.value_FU_OneTask,
            self.payment_Future_min, self.payment_Future_max,
            self.penalty_FU_min, self.penalty_FU_max,
            self.penalty_ESP_min, self.penalty_ESP_max,
            self.cost_ESP, self.ESP_capability,
            self.FU_TaskNum_min, self.FU_TaskNum_max
        )
        best = CDO.select_best_futures_candidate(
            cand, self.cost_ESP, self.ESP_capability, self.FU_TaskNum_min, self.FU_TaskNum_max
        )
        self._last_NI = int(NI)
        return best if best.get("feasible", False) else None

    # ---------------------------
    # RL-style API
    # ---------------------------
    def reset(self) -> np.ndarray:
        """Reset environment states and pick an initial contract."""
        self._ensure()
        self.current_contract = self._pick_contract()
        self._last_total_utility = 0.0
        self._last_NI = 0
        self._last_reputation = 0.0
        self._last_demand = 0
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def step(self, action: int):
        """
        Take one step in the environment.

        Args:
            action (int): 0 = Renew (re-pick contract),
                          1 = Execute (futures exec + spot top-up)

        Returns:
            next_state (np.ndarray[float32, 4])
            reward (float)
            done (bool): always False in this minimal shell
            info (dict): extra info (empty here)
            esp (float): ESP utility obtained in this step (for logging)
            mu  (float): MU utility obtained in this step (for logging)
        """
        self._ensure()

        ESP_u = MU_u = 0.0
        NI = 0
        Rep = 0.0
        demand_total = 0

        if action == 0:
            # ---------- Renew: pick a new contract only ----------
            self.current_contract = self._pick_contract()
            NI = self._last_NI
            reward = SRU.reward(
                action=0, ESP_Utility=0.0, MU_Utility=0.0, Rep=0.0,
                Rew1=self.Rew1, omega6=self.omega6, omega7=self.omega7
            )

        else:
            # ---------- Execute: futures execution + spot top-up ----------
            if self.current_contract is None:
                self.current_contract = self._pick_contract()

            if self.current_contract is not None:
                p = float(self.current_contract["payment"])
                q_f = float(self.current_contract["penalty_FU"])
                contract = np.asarray(self.current_contract["FU_max_tasks"], dtype=int)

                # Dynamic realized demand per FU (can be replaced by real traces)
                actual_demand = np.random.randint(
                    int(self.FU_TaskNum_min), int(self.FU_TaskNum_max) + 1, int(self.Num_FU)
                )
                demand_total = int(np.sum(actual_demand))

                # Futures execution under capacity; returns (+/-) counts for reputation
                ESP_f, MU_f, Fin_f, alloc_vec, Np, Nm = CDO.execute_with_demand(
                    value=self.value_FU_OneTask,
                    payment_fut=p,
                    contract_numTask=contract,
                    actual_demand=actual_demand,
                    capacity_exec=int(self.ESP_capability),
                    penalty_FU=q_f,
                    cost=float(self.cost_ESP)
                )

                # Use remaining capacity in the spot stage (ConSpot)
                remain = int(self.ESP_capability) - int(Fin_f)
                if remain > 0:
                    ESP_s, MU_s, NI_s, Fin_s = TCD.conspot(
                        value=self.value_spot,
                        NumTask=self.NumTask_spot,
                        cost=float(self.cost_ESP),
                        payment_min=float(self.payment_spot_min),
                        payment_max=float(self.payment_spot_max),
                        Num_RB=int(remain)
                    )
                else:
                    ESP_s = MU_s = 0.0
                    NI_s = 0
                    Fin_s = 0

                ESP_u = float(ESP_f + ESP_s)
                MU_u = float(MU_f + MU_s)
                NI = int(NI_s + (self.Num_FU + 1))  # keep style consistent

                # Reputation & reward
                Rep = SRU.reputation(N_plus=int(Np), N_minus=int(Nm), w4=self.w4, w5=self.w5)
                reward = SRU.reward(
                    action=1, ESP_Utility=ESP_u, MU_Utility=MU_u, Rep=Rep,
                    Rew1=self.Rew1, omega6=self.omega6, omega7=self.omega7
                )
            else:
                # No feasible contract -> neutral reward this step
                reward = 0.0

        # Update trackers & build next state
        self._last_total_utility = ESP_u + MU_u
        self._last_NI = int(NI)
        self._last_reputation = float(Rep)
        self._last_demand = int(demand_total)

        next_state = np.array(
            [self._last_demand, self._last_total_utility, self._last_NI, self._last_reputation],
            dtype=np.float32
        )
        done = False
        info: Dict[str, Any] = {}
        return next_state, float(reward), bool(done), info, float(ESP_u), float(MU_u)


# ----------------- Demo settings (replace with real data if needed) -----------------
def demo_settings(Num_FU: int = 30) -> Tuple:
    """
    Generate synthetic demo settings for quick training sanity checks.
    Returns:
        A tuple matching the expected layout in set_settings_tuple(...).
    """
    FU_computer_local = (np.random.randint(10, 15, Num_FU) * 1e8) / 600
    FU_power_trans = np.random.randint(500, 551, Num_FU) * 0.001
    FU_power_local = np.random.randint(450, 501, Num_FU) * 0.001
    ESP_computer_local = (1.5 * 1e12) / 600
    ESP_power_local = 0.7
    task_D = 1.5e6
    Wb = 6e6
    SNR_expected = 250

    time_local = task_D / FU_computer_local
    time_ESP = task_D / ESP_computer_local
    time_trans = task_D / (Wb * np.log2(1 + FU_power_trans * SNR_expected))
    time_save = time_local + time_trans - time_ESP
    energy_save = time_local * FU_power_local - time_trans * FU_power_trans

    V1 = 10.0
    V2 = 10.0
    value_FU_OneTask = V1 * time_save + V2 * energy_save
    cost_hard = 0.001
    cost_ESP = V2 * task_D * ESP_power_local / ESP_computer_local + cost_hard

    payment_Future_max = float(np.max(value_FU_OneTask))
    payment_Future_min = float(np.min(value_FU_OneTask))
    penalty_FU_min, penalty_FU_max = 0.5, 3.0
    penalty_ESP_min, penalty_ESP_max = 0.5, 5.0
    FU_TaskNum_min, FU_TaskNum_max = 7, 12
    ESP_capability = 600
    Num_OU_expected = 20  # placeholder

    return (
        Num_FU, FU_computer_local, FU_power_local, Num_OU_expected,
        FU_TaskNum_min, FU_TaskNum_max, FU_power_trans, cost_ESP,
        ESP_capability, V1, V2, value_FU_OneTask,
        payment_Future_min, payment_Future_max,
        penalty_FU_min, penalty_FU_max, penalty_ESP_min, penalty_ESP_max
    )


# ----------------- Training main (save best model + JSON output) -----------------
def train_and_save(
    episodes: int = 5,
    steps_per_episode: int = 100,
    best_path: str = "ohtrust_ddqn_best.pt",
) -> Dict[str, float]:
    """
    Train a DDQN agent on the Oh-Trust environment and save the best model.

    Args:
        episodes (int): Number of episodes.
        steps_per_episode (int): Steps per episode.
        best_path (str): File path to save the best DDQN model.

    Returns:
        dict: Fixed-schema result of the best episode:
            {
                "method": "ohtrust",
                "ESP_Utility": float,
                "MU_Utility": float,
                "FinishTaskNum": int,
                "NI": int
            }
    """
    env = OhTrustEnv()
    env.set_settings_tuple(demo_settings(Num_FU=30))
    env.set_spot_inputs(
        value_spot=np.asarray(env.value_FU_OneTask, dtype=float),
        NumTask_spot=np.ones(int(env.Num_FU), dtype=int),
        payment_spot_min=float(env.payment_Future_min),
        payment_spot_max=float(env.payment_Future_max),
    )

    agent = SRU.DDQNAgent(state_size=env.state_size, action_size=env.action_size)
    best_ret = -1e18
    best_result: Dict[str, float] = {
        "method": "ohtrust",
        "ESP_Utility": 0.0,
        "MU_Utility": 0.0,
        "FinishTaskNum": 0,
        "NI": 0,
    }

    for _ep in range(episodes):
        s = env.reset()
        ret = 0.0
        ESP_total, MU_total, NI_total, Fin_total = 0.0, 0.0, 0, 0

        for _ in range(steps_per_episode):
            a = agent.act(s)
            s2, r, done, _info, esp, mu = env.step(a)
            agent.remember(s, a, r, s2, done)
            s = s2

            # Aggregate stats
            ret += float(r)
            ESP_total += float(esp)
            MU_total += float(mu)
            NI_total += int(env._last_NI)
            Fin_total += int(env._last_demand)

            agent.replay()
            if done:
                break

        agent.update_target()

        if ret > best_ret:
            best_ret = ret
            agent.save(best_path)
            best_result = {
                "method": "OhTrust",
                "ESP_Utility": float(ESP_total),
                "MU_Utility": float(MU_total),
                "FinishTaskNum": int(Fin_total),
                "NI": int(NI_total),
            }

    # Print once for automated runners / CI logs
    print(best_result)
    return best_result


if __name__ == "__main__":
    train_and_save()
