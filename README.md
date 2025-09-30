# Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update

This repository provides the **official minimal reproducible implementation** of our paper:

> *Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update over Dynamic Edge Networks*  
> Submitted to **IEEE Transactions on Emerging Topics in Computing (TETC), 2025** (under review).

---

## 📂 Project Structure

```
.
├── OhTrust.py          # Main Oh-Trust orchestrator (BiN_CDO + BiN_TCD + SRU_ConR + DDQN)
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
   - Aligns with the *ConFutures baseline*.  

2. **BiN_TCD** (Task-based Spot Trading)  
   - Implements spot-stage settlement and ConSpot baseline.  

3. **SRU_ConR** (Smart Reputation Update & Reward)  
   - Execution-driven reputation monitoring after each long-term contract.  
   - Provides reward function consistent with the paper.  
   - Includes **DDQN agent** implementation.  

4. **OhTrust**  
   - Integrates BiN_CDO, BiN_TCD, and SRU_ConR.  
   - Defines environment states `[demand, total_utility, NI, reputation]`.  
   - Supports actions:  
     - `0 = Renew` (renegotiate contract)  
     - `1 = Execute` (futures + spot execution)  
   - Trains an agent via **DDQN** and saves the best model.  

---

## ⚙️ Requirements

- Python **3.9+**  
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
  ```  

Dependencies:
- numpy>=1.21  
- pandas>=1.3  
- tqdm>=4.62  
- matplotlib>=3.4 (optional, for plotting)  
- scipy>=1.7  

---

## 📊 Dataset

We use the **Chicago Taxi dataset (2013–2016)**, available from the  
[Chicago Data Portal](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew).  

From the raw CSV, we extract:
- **Taxi ID**  
- **Trip Start Timestamp**  
- **Pickup Community Area**  

Then filter **Community Area 77**, and group by `(TaxiID, Date)` to count daily trips (**NumTrips**).  
Rolling statistics over 30 days provide historical demand bounds \(\hat{n}_i\) and \(\tilde{n}_i\).  

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

## 📈 Output

Each run reports **four main indicators**:

- `ESP_Utility`  
- `MU_Utility`  
- `FinishTaskNum`  
- `NI`  

Example output:
```json
{
  "method": "ohtrust",
  "ESP_Utility": 1245.3,
  "MU_Utility": 893.7,
  "FinishTaskNum": 550,
  "NI": 140
}
```

Additional metrics (e.g., PTCT, reputation, task finish rate) are available in code but not included in default outputs.

---

## 📝 Notes

- **α (Pr_alpha)**: For minimal reproducibility, α is simplified here. In practice, α should be **data-driven**, estimated from empirical demand distributions. An extension interface is provided in `Pr.py`.  
- Reputation is updated **after each futures contract execution**, consistent with SRU_ConR.  
- Outputs are aligned with paper definitions: ESP utility, MU utility, finished tasks, and NI count.  
- This repository only releases **core modules and baselines** for peer review reproducibility.  

---

## 📚 Citation

If you use this code, please cite:

```
@article{Qi2025OhTrust,
  title={Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update over Dynamic Edge Networks},
  author={Houyi Qi and ...},
  journal={IEEE Transactions on Emerging Topics in Computing},
  year={2025},
  note={under review}
}
```
