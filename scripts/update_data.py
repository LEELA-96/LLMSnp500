import os
import pandas as pd
from tqdm import tqdm
from supabase import create_client
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bhssymcperznabnnzyin.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJoc3N5bWNwZXp...")
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

# Normalize column names
companies_df.columns = companies_df.columns.str.strip().str.lower()
symbols_df.columns = symbols_df.columns.str.strip().str.lower()

# Ensure required columns
if "symbol" not in symbols_df.columns:
    raise ValueError("No 'symbol' column found in SP500_Symbols_2.csv")
if "name" not in companies_df.columns:
    possible_name_cols = [c for c in companies_df.columns if "name" in c]
    if possible_name_cols:
        companies_df.rename(columns={possible_name_cols[0]: "name"}, inplace=True)
    else:
        raise ValueError("No 'name' column found in Company_S_HQ_1.xlsx")

# Merge data
df = pd.merge(symbols_df, companies_df, on="symbol", how="left")

# Use 'city' as HQ if exists
df["hq"] = df["city"] if "city" in df.columns else None

# Prepare text for embeddings
text_cols = ["symbol", "name"]
if "hq" in df.columns:
    text_cols.append("hq")
df["text"] = df[text_cols].astype(str).agg(" - ".join, axis=1)

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
            "symbol": row.get("symbol"),
            "name": row.get("name"),
            "embedding": row.get("embedding")
        }
        if row.get("hq"):
            record["hq"] = row.get("hq")
        records.append(record)

    res = supabase.table(TABLE_NAME).upsert(records).execute()
    if res.get("error"):
        print("Error inserting batch:", res["error"])

print("🎉 All embeddings uploaded successfully!")
