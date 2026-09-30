import yfinance as yf


# Replacement stocks for the 4 delisted ones
replacements = ["MRO", "APA", "FANG", "OVV"]

START = "2011-03-01"
END   = "2024-09-26"

for ticker in replacements:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START, end=END, progress=False)
    if not df.empty:
        df.to_csv(f"data/raw/{ticker}.csv")
        print(f"  ✓ Saved {ticker} — {len(df)} rows")
    else:
        print(f"  ✗ Failed {ticker}")

print("\nDone! Now you have 50 stocks.")