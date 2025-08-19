import os
import pandas as pd
from tqdm import tqdm
from supabase import create_client
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Debug prints to verify environment variables
print("Supabase URL:", SUPABASE_URL)
print("Supabase Key length:", len(SUPABASE_KEY) if SUPABASE_KEY else "Not Set")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not set. Check environment variables.")

TABLE_NAME = "embeddings"
BATCH_SIZE = 50
EXCEL_FILE = "data/Company_S_HQ_1.xlsx"
CSV_FILE = "data/SP500_Symbols_2.csv"

# ---------------- INIT ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- LOAD DATA ----------------
print("📂 Loading data...")
companies_df = pd.read_excel(EXCEL_FILE)
symbols_df = pd.read_csv(CSV_FILE)

# Merge on 'Symbol'
df = pd.merge(symbols_df, companies_df, on="Symbol", how="left")

# Construct text for embeddings
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
print(f"⬆️ Uploading to Supabase table: {TABLE_NAME}")
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Uploading batches"):
    batch = df.iloc[i:i + BATCH_SIZE]
    records = []
    for _, row in batch.iterrows():
        record = {
            "symbol": row["Symbol"],
            "name": row["Name"],
            "embedding": row["embedding"]
        }
        # Only add HQ if exists
        if "HQ" in row and pd.notnull(row["HQ"]):
            record["hq"] = row["HQ"]
        records.append(record)

    res = supabase.table(TABLE_NAME).upsert(records).execute()
    if res.get("error"):
        print("Error inserting batch:", res["error"])

print("🎉 All embeddings uploaded successfully!")
