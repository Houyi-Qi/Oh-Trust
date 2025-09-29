# Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update

This repository provides a **minimal reproducible implementation** of our paper:

> *Oh-Trust: Overbooking and Hybrid Trading Empowered Resource Scheduling with Smart Reputation Update over Dynamic Edge Networks*  
> Submitted to **IEEE Transactions on Emerging Topics in Computing (TETC), 2025**.

It contains:
1. **Core modules**: Oh-Trust and HybridFS.  
2. **Baselines**: ConSpot, ConFutures, Random.  
3. **Data preprocessing** scripts for the Chicago Taxi dataset.  

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

Then filter **Community Area 77**, and group by `(TaxiID, Date)` to count daily trips (**NumTrips**).  
Rolling statistics over 30 days provide historical demand bounds \(\hat{n}_i\) and \(\tilde{n}_i\).  

---

## Usage

Each baseline and core method can be run separately.  
Example:

```bash
# Run Oh-Trust with 30 MUs
python OhTrust.py --csv data/chicago_taxi.csv --num_mu 30

# Run HybridFS
python HybridFS.py --csv data/chicago_taxi.csv --num_mu 30

# Run ConSpot
python ConSpot.py --csv data/chicago_taxi.csv --num_mu 30

# Run ConFutures
python ConFutures.py --csv data/chicago_taxi.csv --num_mu 30

# Run Random
python Random.py --csv data/chicago_taxi.csv --num_mu 30
```

---

## Output

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

---

## Notes

- **Oh-Trust** integrates futures–spot trading, overbooking, and contract update mechanism.  
- We only release **core modules** and **baseline implementations** here for reproducibility.  
- Extra indicators (e.g., PTCT, reputation, task finish rate) are available in code but omitted in default outputs.  
- This repo is intended to support **peer review** of the TETC submission.  

---


