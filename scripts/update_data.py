# scripts/update_data.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
from supabase import create_client
from sentence_transformers import SentenceTransformer

# ----------------------------
# Supabase config (use environment variables)
# ----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL") or "https://bhssymcperznabnnzyin.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJoc3N5bWNwZXp..."
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# File paths
# ----------------------------
COMPANY_FILE = "data/Company_S_HQ_1.xlsx"
SYMBOLS_FILE = "data/SP500_Symbols_2.csv"

# ----------------------------
# Table names
# ----------------------------
COMPANY_TABLE = "company_metadata"
STOCK_TABLE = "stock_data"
EMBEDDINGS_TABLE = "embeddings"

# ----------------------------
# Load company & symbol data
# ----------------------------
companies_df = pd.read_excel(COMPANY_FILE)
symbols_df = pd.read_csv(SYMBOLS_FILE)

# ----------------------------
# Update company metadata
# ----------------------------
company_records = []
for _, row in companies_df.iterrows():
    hq = row.get("Headquarters") or row.get("City") or "Unknown"
    record = {
        "symbol": row.get("Symbol"),
        "company_name": row.get("Name"),
        "hq": hq
    }
    company_records.append(record)

for rec in company_records:
    try:
        supabase.table(COMPANY_TABLE).upsert(rec).execute()
    except Exception as e:
        print(f"Error inserting company {rec['symbol']}: {e}")

print("✅ Company metadata updated.")

# ----------------------------
# Fetch past 5 years stock data
# ----------------------------
start_date = datetime.now() - timedelta(days=5*365)
end_date = datetime.now()
all_stock_data = []

for symbol in tqdm(symbols_df["Symbol"], desc="Fetching stock data"):
    try:
        df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"),
                         end=end_date.strftime("%Y-%m-%d"), progress=False)
        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "date",
            "Open": "open",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Volume": "volume"
        }, inplace=True)
        df["symbol"] = symbol
        df = df[["symbol", "date", "open", "close", "high", "low", "volume"]]
        all_stock_data.append(df)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

if all_stock_data:
    stock_df = pd.concat(all_stock_data)
    stock_records = stock_df.to_dict(orient="records")
    for i in range(0, len(stock_records), 100):
        batch = stock_records[i:i+100]
        supabase.table(STOCK_TABLE).upsert(batch).execute()

print("✅ Stock data updated.")

# ----------------------------
# Generate embeddings
# ----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_records = []

for _, row in companies_df.iterrows():
    hq = row.get("Headquarters") or row.get("City") or "Unknown"
    text = f"{row.get('Symbol')} - {row.get('Name')} - {hq}"
    embedding_records.append({
        "symbol": row.get("Symbol"),
        "name": row.get("Name"),
        "hq": hq,
        "text": text
    })

# Batch embedding upload
batch_size = 50
for i in tqdm(range(0, len(embedding_records), batch_size), desc="Encoding & uploading embeddings"):
    batch = embedding_records[i:i+batch_size]
    texts = [x["text"] for x in batch]
    embeddings = model.encode(texts).tolist()

    records_to_upload = []
    for j, x in enumerate(batch):
        record = {
            "symbol": x["symbol"],
            "name": x["name"],
            "text": x["text"],
            "embedding": embeddings[j]
        }
        # Include HQ only if column exists
        try:
            supabase.table(EMBEDDINGS_TABLE).select("hq").execute()
            record["hq"] = x["hq"]
        except:
            pass
        records_to_upload.append(record)

    try:
        supabase.table(EMBEDDINGS_TABLE).upsert(records_to_upload).execute()
    except Exception as e:
        print(f"Error uploading embeddings batch: {e}")

print("✅ Embeddings generated and uploaded successfully.")
