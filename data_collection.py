import yfinance as yf

import os

# Indian Energy Sector Stocks (NSE listed - Yahoo Finance uses .NS suffix)
stocks = [
    # Oil & Gas Exploration
    "ONGC.NS",       # Oil and Natural Gas Corporation
    "OIL.NS",        # Oil India Limited
    "CAIRN.NS",      # Cairn India

    # Refining & Marketing
    "IOC.NS",        # Indian Oil Corporation
    "BPCL.NS",       # Bharat Petroleum
    "HPCL.NS",       # Hindustan Petroleum
    "MRPL.NS",       # Mangalore Refinery

    # Power Generation
    "NTPC.NS",       # NTPC Limited
    "POWERGRID.NS",  # Power Grid Corporation
    "TATAPOWER.NS",  # Tata Power
    "ADANIPOWER.NS", # Adani Power
    "CESC.NS",       # CESC Limited
    "TORNTPOWER.NS", # Torrent Power
    "JSWENERGY.NS",  # JSW Energy
    "NHPC.NS",       # NHPC Limited
    "SJVN.NS",       # SJVN Limited

    # Renewable Energy
    "ADANIGREEN.NS", # Adani Green Energy
    "ATGL.NS",       # Adani Total Gas
    "IGL.NS",        # Indraprastha Gas
    "MGL.NS",        # Mahanagar Gas
    "GUJGASLTD.NS",  # Gujarat Gas

    # Coal & Mining
    "COALINDIA.NS",  # Coal India
    "NMDC.NS",       # NMDC Limited

    # Petrochemicals
    "RELIANCE.NS",   # Reliance Industries
    "GAIL.NS",       # GAIL India
    "PETRONET.NS",   # Petronet LNG
    "GSPL.NS",       # Gujarat State Petronet

    # Energy Equipment & Services
    "BEL.NS",        # Bharat Electronics
    "BHEL.NS",       # Bharat Heavy Electricals
    "THERMAX.NS",    # Thermax Limited
]

START = "2011-03-01"
END   = "2024-09-26"

os.makedirs("data/raw_indian3", exist_ok=True)

success = 0
failed = []

for ticker in stocks:
    print(f"Downloading {ticker}...")
    try:
        df = yf.download(ticker, start=START, end=END, progress=False)
        if not df.empty:
            filename = ticker.replace(".", "_").replace("-", "_")
            df.to_csv(f"data/raw_indian3/{filename}.csv")
            print(f"  ✓ Saved {ticker} — {len(df)} rows")
            success += 1
        else:
            print(f"  ✗ No data for {ticker}")
            failed.append(ticker)
    except Exception as e:
        print(f"  ✗ Error {ticker}: {e}")
        failed.append(ticker)

print(f"\n ✓ Successfully downloaded: {success} stocks")
print(f"✗ Failed: {len(failed)} stocks → {failed}")