# GCN-LSTM+RSI: Hybrid Deep Learning Model for Energy Stock Price Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Published](https://img.shields.io/badge/Published-IJSART%202025-green.svg)](https://ijsart.com)

## Overview

A novel hybrid deep learning architecture that combines **Graph Convolutional Networks (GCN)** with **attention-enhanced LSTM** for predicting energy sector stock prices. The model extends the base paper (Amiri et al., IEEE Access 2025) by integrating the **Relative Strength Index (RSI)** as an additional momentum feature.

**Target Stock:** XOM (Exxon Mobil Corporation)  
**Dataset:** 24 US Energy Sector Stocks | June 2011 – September 2024  
**Best Result:** R² = 0.6403 | RMSE = $4.14

---

## Architecture

```
Raw Price Data (24 stocks)
        ↓
RSI Feature Engineering (14-period)
        ↓
DTW-Based Graph Construction (threshold=0.7, 368 edges)
        ↓
GCN Layer 1 (in=2, out=32)
        ↓
GCN Layer 2 (out=64)
        ↓
2-Layer LSTM (64 units, dropout=0.3)
        ↓
Multi-Head Attention (4 heads)
        ↓
Predicted Opening Price
```

**Total Parameters:** 85,601

---

## Results

| Stock | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|----|
| XOM (Exxon Mobil) | 17.11 | $4.14 | $3.46 | **0.6403** |
| COP (ConocoPhillips) | 30.78 | $5.55 | $4.46 | **0.1453** |
| CVX (Chevron) | 254.28 | $15.95 | $14.54 | -4.13 |

**Test Period:** August 2023 – September 2024 (263 unseen trading days)

**Base Paper (Amiri et al. 2025):** XOM R² = 0.815 with 50 stocks  
**Our Model:** XOM R² = 0.6403 with 24 stocks + RSI feature

---

## Stocks Used (24 US Energy Sector)

| Sub-Sector | Tickers |
|---|---|
| Major Integrated Oil | XOM, CVX, SHEL, TTE, BP, E |
| E&P | COP, EOG, DVN, CNQ |
| Oilfield Services | SLB, HAL, BKR |
| Midstream | ENB, EPD, WMB, OKE, LNG, TRP |
| Refining | VLO, MPC |
| International | EC, IMO, SU |

---

## Project Structure

```
gcn-lstm-rsi-stock-prediction/
├── data_collection.py              # Module 1: Download stock data via yfinance
├── module2_data_preprocessing.py   # Module 2: RSI computation + normalization
├── module3_graph_construction.py   # Module 3: DTW-based graph construction
├── module5_GCN_feature_Extraction.py # Module 4: GCN feature extraction
├── gcn_lstm_rsi_complete.py        # Complete end-to-end pipeline
├── requirements.txt                # Dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yokeshkumar/gcn-lstm-rsi-stock-prediction.git
cd gcn-lstm-rsi-stock-prediction

python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## Usage

**Step 1: Download stock data**
```bash
python data_collection.py
```

**Step 2: Preprocess and compute RSI**
```bash
python module2_data_preprocessing.py
```

**Step 3: Build DTW graph**
```bash
python module3_graph_construction.py
```

**Step 4: Run complete pipeline (recommended on GPU)**
```bash
python gcn_lstm_rsi_complete.py
```

---

## Requirements

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
dtaidistance>=2.3.0
```

---

## Key Findings

- **RSI contribution:** Adding RSI improved R² by 0.219 points (52% relative improvement) over price-only baseline
- **Graph density:** 368 edges formed among 24 stocks with DTW threshold 0.7 (66.67% density)
- **Distribution shift:** COVID-19 caused energy stock price regime change in 2021-2022, limiting model generalization on standard 70/20/10 splits. Using 85/7/8 split resolved this.
- **Graph size matters:** Base paper used 50 stocks (R²=0.815); our 24-stock graph achieved R²=0.6403

---

## Limitations

- Static graph — DTW adjacency computed once from training data
- No macroeconomic features (crude oil price, USD/EUR rate)
- COVID-19 price regime change creates distribution shift
- 24 stocks vs 50 in base paper reduces graph richness

---

## Future Work

- Expand to 50 stocks matching base paper
- Add MACD, Bollinger Bands as additional features
- Integrate sentiment analysis for black swan event detection
- Dynamic graph updated monthly
- Deploy as REST API for real-time predictions

---

## Publication

This project is published in the **International Journal of Science and Research Technology (IJSART)**  
Paper ID: IJSARTV12I3104734

---

## References

B. Amiri, A. Haddadi, and K. F. Mojdehi, "A Novel Hybrid GCN-LSTM Algorithm for Energy Stock Price Prediction," *IEEE Access*, vol. 13, pp. 24815-24832, 2025.

---

## Author

**Yokeshkumar Elangovan**  
B.E. Computer Science and Engineering  
Mahendra Institute of Engineering and Technology, Namakkal  
📧 yokesh1651@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/yokesh-kumar-8b0672298)
