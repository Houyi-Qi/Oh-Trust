# Oh-Trust: Overbooking and Hybrid Trading-Empowered Resource Scheduling with Smart Reputation Update

This repository provides a **minimal reproducible implementation** of our paper:

> *Oh-Trust: Overbooking and Hybrid Trading-Empowered Resource Scheduling With Smart Reputation Update Over Dynamic Edge Networks*
>
> **IEEE Transactions on Emerging Topics in Computing (TETC), 2025**
>
> DOI: [10.1109/TETC.2025.3625379](https://doi.org/10.1109/TETC.2025.3625379)

---

## 📂 Project Structure

~~~text
.
├── OhTrust.py          # Main workflow, training, and comparisons
├── BiN_CDO.py          # Futures contract design
├── BiN_TCD.py          # Contract execution and spot trading
├── SRU_ConR.py         # Reputation, reward function, and DDQN
├── ConFutures.py       # Pure futures baseline
├── ConSpot.py          # Pure spot baseline
├── HybridFS.py         # Hybrid futures and spot baseline
├── Random.py           # Random allocation baseline
├── Utility.py          # Settlement, utility, and risk calculations
├── Pr.py               # Demand scenarios and probability calculations
├── data_utils.py       # Data generation and preprocessing
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
~~~

---

## 🔑 Core Modules

1. **BiN_CDO: Futures Contract Design**
   Evaluates contracts using historical demand and selects feasible candidates under price, risk, and overbooking constraints.

2. **BiN_TCD: Execution and Spot Trading**
   Executes futures contracts, handles buyer defaults and volunteer compensation, and uses remaining capacity to serve unmet demand.

3. **SRU_ConR: Reputation and Renewal**
   Updates reputation from completed trades and uses Double DQN to choose between execution and renewal.

4. **OhTrust: Integrated Workflow**
   Combines these modules with state `[demand, total_utility, NI, reputation]` and actions `0 = Renew` and `1 = Execute`. Renewal uses observed history only and does not consume a trading slot.

---

## ⚙️ Requirements

- Python **3.10+**; CPU execution is supported.
- Main dependencies: NumPy, SciPy, pandas, PyTorch, and pytest.

~~~bash
python -m pip install -r requirements.txt
~~~

---

## 📊 Dataset

The paper uses **Chicago Taxi Trips (2013–2016)**, available from the [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew).

Preprocessing reads `Taxi ID`, `Trip Start Timestamp`, and `Pickup Community Area`, filters community area **77**, and counts daily trips by taxi. Historical windows estimate demand bounds while excluding the current trade.

By default, the code uses synthetic demand with a fixed random seed, requiring no dataset download. Local CSV input is supported through `--csv`.

---

## 🚀 Usage

### Run Oh-Trust

~~~bash
python OhTrust.py
~~~

Trains DDQN, selects a model using validation returns, and evaluates it on a separate test trajectory. The default checkpoint is `ohtrust_ddqn_best.pt`.

### Run Baselines

~~~bash
python Random.py
python ConFutures.py
python ConSpot.py
python HybridFS.py
~~~

### Compare All Methods on the Same Demand Trajectory

~~~bash
python OhTrust.py --compare --seed 7 --slots 12 --episodes 3
~~~

To use a local raw taxi dataset:

~~~bash
python OhTrust.py --compare --csv taxi.csv --raw-taxi --num-fb 3 --warmup 30
~~~

---

## 📈 Output

- `ESP_Utility`: Realized server utility.
- `MU_Utility`: Realized buyer utility.
- `FinishTaskNum`: Tasks actually completed through futures and spot trading.
- `NI`: Offline contract search and online spot negotiation counts.

Results are saved to `results/compare.json` by default; use `--output` to change the path. Outputs also include defaults, volunteer compensation, renewals, and settlement details. NI is an implementation-level proxy for negotiation overhead.

---

## 📝 Notes

- Updates unify settlement and probability estimation, fix residual demand allocation, history leakage, and DDQN renewal, and give all baselines the same demand traces and metrics.
- Risks use historical samples. Task-level settlement separates buyer default penalties from volunteer compensation. These adapt the paper's mechanisms; large contract spaces use a heuristic search.
- Defaults are `tau = 0.1` and `0.3` for all three risk thresholds. Synthetic capacity, valuations, and other demo parameters do not represent the paper's experimental configuration.
- Original filenames and command-line entry points are retained. Some internal APIs now require explicit demand, capacity, or related inputs.
- This release supports checking core mechanisms; it does not reproduce all paper figures or establish a performance advantage.

---

## 📚 Citation

If you use this code, please cite:

~~~bibtex
@article{qi2025ohtrust,
  author={Qi, Houyi and Liwang, Minghui and Fu, Liqun and Wang, Xianbin and Dai, Huaiyu and Xia, Xiaoyu},
  journal={IEEE Transactions on Emerging Topics in Computing},
  title={Oh-Trust: Overbooking and Hybrid Trading-Empowered Resource Scheduling With Smart Reputation Update Over Dynamic Edge Networks},
  year={2025},
  volume={13},
  number={4},
  pages={1553-1564},
  doi={10.1109/TETC.2025.3625379}
}
~~~
