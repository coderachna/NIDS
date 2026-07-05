# Network Intrusion Detection System (NIDS) — ML on Network Traffic

A machine learning system that classifies network traffic as **Normal** or one of three attack types — **DoS**, **Port Scan**, or **Brute Force** — using the CICIDS2017 dataset (2.8M+ labeled network flow records).

## Problem

Traditional intrusion detection relies on hand-written rules (signature-based detection), which can't keep up with the volume and evolving nature of network attacks — a rule only catches what it was explicitly written to catch. This project instead trains a model on real, labeled attack traffic so it learns the underlying statistical *behavior* of each attack type, allowing it to generalize better to traffic it wasn't explicitly told about.

## Dataset

- **Source:** [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity)
- **Size:** 2,830,743 labeled network flow records across 8 CSV files (Monday–Friday capture)
- **Features:** ~78 flow-level statistics per record (packet size, duration, flag counts, etc.)
- **Target classes (after mapping):** Normal, DoS (incl. DDoS/Hulk/GoldenEye/slowloris/Slowhttptest), PortScan, BruteForce (FTP-Patator, SSH-Patator)

## Pipeline

1. **Load & combine** all 8 raw CSVs into a single DataFrame
2. **Clean** — strip column whitespace, fix character encoding, map 15 raw label strings into 4 target classes, drop out-of-scope attack types (Bot, Web Attacks, Infiltration, Heartbleed — outside project scope)
3. **Handle infinities/NaNs** — caused by division-by-zero in rate-based features (`Flow Bytes/s`, `Flow Packets/s`) on near-instant flows; ~0.1% of rows dropped
4. **Encode & scale** — label-encode target classes, scale features with `StandardScaler` (fit on train only, to avoid data leakage)
5. **Stratified train/test split** (80/20) — preserves class proportions given severe imbalance (BruteForce is only ~0.5% of data)
6. **Train & compare** — Decision Tree and Random Forest classifiers
7. **Evaluate** — per-class precision/recall/F1 and confusion matrix (not just accuracy, which is misleading on imbalanced data)
8. **Dashboard** — Flask app with two modes: (a) single-flow prediction, pulling a random row from the held-out test set and showing predicted class + confidence + ground truth, and (b) batch classification via CSV upload, returning predictions and a per-class breakdown for every row

## Results

Both models performed extremely well on the held-out test set:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 1.00 | 1.00 | 1.00 |
| DoS | 1.00 | 1.00 | 1.00 |
| PortScan | 0.99 | 0.99 | 0.99 |
| BruteForce | 1.00 | 1.00 | 1.00 |

**Top predictive features** (Random Forest `feature_importances_`): packet-size statistics dominate — Avg Bwd Segment Size, Packet Length Std, Max Packet Length, and Packet Length Variance were the top 4, followed by Destination Port. This aligns with intuition: DoS/port-scan/brute-force traffic tends to be small, repetitive, and low-payload, unlike varied normal browsing/download traffic.

*Note: near-perfect scores on a single benchmark dataset are encouraging but not conclusive — this is a known characteristic of CICIDS2017's attack simulations being relatively distinct from normal traffic. A production system would need validation on live, real-world traffic.*

## Tech Stack

- **Data processing:** Python, pandas, NumPy
- **Modeling:** scikit-learn (DecisionTreeClassifier, RandomForestClassifier)
- **Dashboard:** Flask
- **Model persistence:** joblib

## Project Structure

\```
NIDS-CICIDS2017/
├── data/                    # raw + cleaned CSVs (excluded from git — see .gitignore)
├── src/
│   ├── 01_load_combine.py   # loads, cleans, and combines raw CSVs
│   └── 02_train_model.py    # trains/evaluates Decision Tree & Random Forest, saves .pkl files
├── app/
│   ├── app.py                # Flask dashboard (single + batch prediction)
│   └── templates/
├── rf_model.pkl              # trained model (excluded from git — see below)
├── scaler.pkl                # fitted StandardScaler (excluded from git)
├── label_encoder.pkl         # fitted LabelEncoder (excluded from git)
├── .gitignore
└── README.md
\```

## Running the Dashboard

\```bash
pip install flask pandas numpy scikit-learn joblib
python app/app.py
\```
Then open `http://127.0.0.1:5000` in a browser.

**Note:** `rf_model.pkl`, `scaler.pkl`, and `label_encoder.pkl` are excluded from this repo via `.gitignore` (kept lightweight, standard practice for trained model binaries). To run the dashboard yourself, first run `src/01_load_combine.py` then `src/02_train_model.py` to regenerate these files locally.
