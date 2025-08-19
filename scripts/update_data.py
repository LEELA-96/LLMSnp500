import pandas as pd
import numpy as np
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# -----------------------------
# Config
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "embeddings"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load company & symbols data
# -----------------------------
companies_df = pd.read_excel("Company_S_HQ_1.xlsx")
symbols_df = pd.read_csv("SP500_Symbols_2.csv")

# Standardize column names
companies_df.columns = [c.strip().lower() for c in companies_df.columns]
symbols_df.columns = [c.strip().lower() for c in symbols_df.columns]

# Merge on symbol
df = pd.merge(symbols_df, companies_df, on="symbol", how="inner")

print(f"✅ Merged data shape: {df.shape}")

# -----------------------------
# Generate embeddings
# -----------------------------
rows_to_insert = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding rows"):
    text = f"{row['symbol']} {row['name']} {row['sector']}"
    embedding = model.encode(text).tolist()
    rows_to_insert.append({
        "symbol": row["symbol"],
        "name": row["name"],
        "sector": row["sector"],
        "embedding": embedding
    })

print(f"✅ Generated embeddings for {len(rows_to_insert)} rows")

# -----------------------------
# Insert into Supabase
# -----------------------------
BATCH_SIZE = 100
for i in range(0, len(rows_to_insert), BATCH_SIZE):
    batch = rows_to_insert[i:i+BATCH_SIZE]
    print(f"⬆️ Inserting batch {i//BATCH_SIZE+1} ({len(batch)} rows)")
    res = supabase.table(TABLE_NAME).insert(batch).execute()
    if hasattr(res, "error") and res.error:
        print("❌ Error inserting:", res.error)
    else:
        print("✅ Batch inserted successfully")
