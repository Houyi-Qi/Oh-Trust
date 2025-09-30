# Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update

This repository contains the official reproducible implementation of our paper:  

> *Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update over Dynamic Edge Networks*  
> IEEE Transactions on Emerging Topics in Computing (under review, 2025).  

---

## 📂 Project Structure

```
.
├── OhTrust.py          # Main Oh-Trust RL orchestrator (BiN_CDO + BiN_TCD + SRU_ConR + DDQN)
├── BiN_CDO.py          # Futures trading (ConFutures baseline, contract optimization)
├── BiN_TCD.py          # Spot trading (ConSpot baseline)
├── SRU_ConR.py         # Reputation update + reward function + DDQN agent
├── ConFutures.py       # Pure Futures baseline
├── ConSpot.py          # Pure Spot baseline
├── HybridFS.py         # Hybrid Futures+Spot baseline
├── Random.py           # Random allocation baseline
├── Utility.py          # Utility functions consistent with paper equations
├── Pr.py               # Probability-related calculations (α, β, φ)
└── README.md           # Project documentation (this file)
```

---

## 🔑 Core Modules

1. **BiN_CDO** (Contract-based Futures Trading)  
   - Implements candidate generation, best contract selection, and execution under demand uncertainty.  
   - Aligns with *ConFutures baseline*.

2. **BiN_TCD** (Task-based Spot Trading)  
   - Implements spot-stage settlement and ConSpot baseline.  

3. **SRU_ConR** (Smart Reputation Update & Reward)  
   - Execution-driven reputation monitoring after each contract fulfillment.  
   - Provides reward function consistent with the paper.  
   - Includes DDQN agent implementation.

4. **OhTrust**  
   - Integrates BiN_CDO, BiN_TCD, and SRU_ConR.  
   - Defines environment states `[demand, total_utility, NI, reputation]`.  
   - Supports actions:  
     - `0 = Renew` (renegotiate contract),  
     - `1 = Execute` (futures + spot execution).  
   - Trains an agent via **DDQN** and saves the best model.

---

## ⚙️ Requirements

- Python **3.9+**  
- Dependencies:
  - numpy>=1.21  
  - pandas>=1.3  
  - tqdm>=4.62  
  - matplotlib>=3.4  
  - scipy>=1.7  

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Baselines

The following baselines are included for fair comparison:

- **Random** (`Random.py`)  
- **ConFutures** (`ConFutures.py`)  
- **ConSpot** (`ConSpot.py`)  
- **HybridFS** (`HybridFS.py`)  

All baselines output consistent evaluation metrics:

```json
{
  "method": "baseline_name",
  "ESP_Utility": 1245.3,
  "MU_Utility": 893.7,
  "FinishTaskNum": 550,
  "NI": 140
}
```

---

## 🚀 Usage

### Run Oh-Trust training
```bash
python OhTrust.py
```

This will:
- Train DDQN agent on the Oh-Trust environment.  
- Save the best model to:
  ```
  ohtrust_ddqn_best.pt
  ```

### Run baselines
```bash
python Random.py
python ConFutures.py
python ConSpot.py
python HybridFS.py
```

---

## 📑 Notes

- **α (Pr_alpha)**: In this repo, α is simplified for minimal reproducibility. In practice, α should be **data-driven**, estimated from empirical demand distributions. An interface is provided in `Pr.py` to extend α to dataset-based estimation.  
- Reputation is updated **after each long-term futures contract execution**, consistent with the SRU_ConR mechanism.  
- Outputs are aligned with paper definitions: ESP utility, MU utility, finished tasks, and NI count.
