import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

# ── Configuration ──────────────────────────────────────────────────
DATA_DIR        = 'data/raw_indian/'
PROCESSED_DIR   = 'data/processed/'
SCALERS_DIR     = 'data/scalers/'
START_DATE      = '2011-03-01'
END_DATE        = '2024-09-26'
RSI_PERIOD      = 14

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR,   exist_ok=True)

# ── Step 1: Load stock CSV ─────────────────────────────────────────
def load_stock(filepath):
    df = pd.read_csv(filepath, skiprows=3, header=None,
                     names=['Date','Close','High','Low','Open','Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open']].astype(float)          # ✅ Fixed: Open not Close
    return df

# ── Step 2: Temporal alignment ─────────────────────────────────────
def align_dates(df):
    full_range = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    df = df.reindex(full_range)
    return df

# ── Step 3: Forward fill missing values ───────────────────────────
def fill_missing(df):
    df = df.ffill()
    df = df.bfill()
    return df

# ── Step 4: RSI Calculation ────────────────────────────────────────
def compute_rsi(series, period=RSI_PERIOD):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / (avg_loss + 1e-10)      # ✅ Fixed: no division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ── Step 5: Normalize ──────────────────────────────────────────────
def normalize(df, name):
    price_scaler = MinMaxScaler()
    rsi_scaler   = MinMaxScaler()
    df['Open_scaled'] = price_scaler.fit_transform(df[['Open']])  # ✅ Fixed
    df['RSI_scaled']  = rsi_scaler.fit_transform(df[['RSI']])
    with open(f"{SCALERS_DIR}{name}_price_scaler.pkl", 'wb') as f:
        pickle.dump(price_scaler, f)
    with open(f"{SCALERS_DIR}{name}_rsi_scaler.pkl", 'wb') as f:
        pickle.dump(rsi_scaler, f)
    return df

# ── Main Pipeline ──────────────────────────────────────────────────
def preprocess_all():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"Found {len(files)} stock files\n")
    success, failed = 0, []

    for file in sorted(files):
        name     = file.replace('.csv', '')
        filepath = DATA_DIR + file
        try:
            print(f"Processing {name}...")
            df = load_stock(filepath)
            print(f"  Loaded: {df.shape[0]} rows")
            df = align_dates(df)
            df = fill_missing(df)
            df['RSI'] = compute_rsi(df['Open'])           # ✅ Fixed
            df = df.iloc[RSI_PERIOD:]
            df = df.dropna()
            df = normalize(df, name)
            df[['Open_scaled', 'RSI_scaled']].to_csv(
                f"{PROCESSED_DIR}{name}_processed.csv"
            )
            print(f"  ✓ Saved {name}_processed.csv\n")
            success += 1
        except Exception as e:
            print(f"  ✗ Failed {name}: {e}\n")
            failed.append(name)

    print(f"{'='*50}")
    print(f"✓ Successfully processed: {success} stocks")
    print(f"✗ Failed: {len(failed)} → {failed}")

if __name__ == "__main__":
    preprocess_all()