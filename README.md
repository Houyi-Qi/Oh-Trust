# Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update

This repository provides a **minimal reproducible implementation** of our paper:

> *Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update over Dynamic Edge Networks*  
> IEEE Transactions on Emerging Topics in Computing, 2025.

It contains:
1. **Core methods**: Oh-Trust and HybridFS.  
2. **Baselines**: ConSpot, ConFutures, Random.  
3. **Data preprocessing** scripts for the real-world Chicago Taxi dataset.  

---

## Requirements

We recommend Python **3.9+**.  
Install dependencies:

```bash
pip install -r requirements.txt
```

Minimal requirements:
- numpy  
- pandas  
- tqdm  
- matplotlib (optional, for plotting)  

---

## Data Preparation

We use the **Chicago Taxi dataset (2013–2016)**.  
Download it from [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew).

From the raw CSV, we extract:
- **Taxi ID**  
- **Trip Start Timestamp**  
- **Pickup Community Area**  

Then we filter **Community Area 77**, and group by `(TaxiID, Date)` to count daily trips (**NumTrips**).  
Rolling statistics over 30 days provide historical demand bounds \(\hat{n}_i\) and \(\tilde{n}_i\).  

---

## Usage

Run experiments via `runner.py`:

```bash
# Run Oh-Trust with 30 MUs
python runner.py --csv data/chicago_taxi.csv --method ohtrust --num_mu 30

# Run HybridFS baseline
python runner.py --csv data/chicago_taxi.csv --method hybridfs --num_mu 30

# Run ConSpot baseline
python runner.py --csv data/chicago_taxi.csv --method conspot --num_mu 30

# Run ConFutures baseline
python runner.py --csv data/chicago_taxi.csv --method confutures --num_mu 30

# Run Random baseline
python runner.py --csv data/chicago_taxi.csv --method random --num_mu 30
```

---

## Output

Each run prints **four core indicators**:

- `ESP_Utility`  
- `MU_Utility`  
- `FinishTaskNum`  
- `NI`  

Example:
```json
{
  "method": "ohtrust",
  "ESP_Utility": 1245.3,
  "MU_Utility": 893.7,
  "FinishTaskNum": 550,
  "NI": 140
}
```

---

## Notes

- **Oh-Trust** integrates futures–spot trading, overbooking, and DDQN-based contract updates.  
- By default, only **four main indicators** are shown. Extra metrics (e.g., PTCT, reputation, task finish rate) are available in the code but hidden for clarity.  
- This repo is intended for reproducibility of paper results, not for deployment.  

---

## Citation

If you use this code, please cite:

```
@article{Qi2025OhTrust,
  title={Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update over Dynamic Edge Networks},
  author={Houyi Qi and ...},
  journal={IEEE Transactions on Emerging Topics in Computing},
  year={2025}
}
```
