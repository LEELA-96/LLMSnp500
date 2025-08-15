import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1️⃣ Supabase setup
# -----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# 2️⃣ File paths
# -----------------------------
symbols_file = "data/SP500_Symbols_2.csv"
company_file = "data/Company_S&HQS (1).xlsx"

# -----------------------------
# 3️⃣ Load files
# -----------------------------
symbols_df = pd.read_csv(symbols_file)
company_df = pd.read_excel(company_file)

# Strip column names
company_df.columns = company_df.columns.str.strip()
symbols_df.columns = symbols_df.columns.str.strip()

# -----------------------------
# 4️⃣ Insert/update company metadata
# -----------------------------
print("🔹 Updating company metadata...")

for _, row in company_df.iterrows():
    try:
        supabase.table("company_metadata").upsert({
            "symbol": row["Symbol"],
            "company_name": row["Company Name"],
            "headquarters": row.get("City", "Unknown")  # use City as HQ
        }).execute()
    except Exception as e:
        print(f"Error inserting company {row['Symbol']}: {e}")

print("✅ Company metadata updated.")

# -----------------------------
# 5️⃣ Fetch historical stock data & embeddings
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')  # fast & free

for _, row in tqdm(symbols_df.iterrows(), total=len(symbols_df), desc="Processing symbols"):
    symbol = row["Symbol"]

    # Check latest date in Supabase
    try:
        res = supabase.table("stock_data").select("date").eq("symbol", symbol).order("date", desc=True).limit(1).execute()
        if res.data:
            last_date = pd.to_datetime(res.data[0]['date'])
            start_date = last_date + timedelta(days=1)
        else:
            start_date = datetime.now() - timedelta(days=5*365)  # last 5 years
    except Exception as e:
        print(f"Error checking last date for {symbol}: {e}")
        start_date = datetime.now() - timedelta(days=5*365)

    end_date = datetime.now()
    if start_date > end_date:
        continue  # already up to date

    # Download stock data
    try:
        df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        continue

    if df.empty:
        continue

    df.reset_index(inplace=True)
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    for _, r in df.iterrows():
        try:
            date_val = pd.to_datetime(r["date"])
            date_str = date_val.strftime("%Y-%m-%d")

            # Insert stock data
            supabase.table("stock_data").upsert({
                "symbol": symbol,
                "date": date_str,
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": int(r["volume"])
            }).execute()

            # Generate embedding
            text_for_embedding = f"{symbol} {date_str} open:{r['open']} close:{r['close']} high:{r['high']} low:{r['low']} volume:{r['volume']}"
            embedding = model.encode(text_for_embedding).tolist()

            # Insert embedding
            supabase.table("stock_embeddings").upsert({
                "symbol": symbol,
                "date": date_str,
                "embedding": embedding
            }).execute()

        except Exception as e:
            print(f"Error processing {symbol} on {r['date']}: {e}")

print("✅ Stock data and embeddings updated successfully!")
