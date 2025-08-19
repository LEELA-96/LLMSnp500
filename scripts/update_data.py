import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from supabase import create_client
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

TABLE_NAME = "embeddings"
BATCH_SIZE = 50
EXCEL_FILE = "data/Company_S_HQ_1.xlsx"   # ✅ fixed path
CSV_FILE = "data/SP500_Symbols_2.csv"

# ---------------- INIT ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- LOAD DATA ----------------
print("📂 Loading data...")
companies_df = pd.read_excel(EXCEL_FILE)
symbols_df = pd.read_csv(CSV_FILE)

# Merge company and symbol data
df = pd.merge(symbols_df, companies_df, on="Symbol", how="left")

# If 'HQ' column missing, skip it safely
text_columns = ["Symbol", "Name"]
if "HQ" in df.columns:
    text_columns.append("HQ")

df["text"] = df[text_columns].astype(str).agg(" - ".join, axis=1)

# ---------------- EMBEDDINGS ----------------
print(f"🧾 Encoding {len(df)} rows...")
embeddings = []
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Encoding rows"):
    batch_texts = df["text"].iloc[i:i + BATCH_SIZE].tolist()
    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True).tolist()
    embeddings.extend(batch_embeddings)

df["embedding"] = embeddings
print(f"✅ Generated embeddings for {len(df)} rows")

# ---------------- UPSERT TO SUPABASE ----------------
print(f"⬆️ Inserting into Supabase table: {TABLE_NAME}")

for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Uploading batches"):
    batch = df.iloc[i:i + BATCH_SIZE]
    records = []
    for _, row in batch.iterrows():
        record = {
            "symbol": row["Symbol"],
            "name": row["Name"],
            "embedding": row["embedding"]
        }
        if "HQ" in row:  # only if HQ column exists
            record["hq"] = row["HQ"]
        records.append(record)

    res = supabase.table(TABLE_NAME).upsert(records).execute()

print("🎉 Done! All embeddings uploaded to Supabase.")
