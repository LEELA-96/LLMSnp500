import os
import pandas as pd
from tqdm import tqdm
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# ---------------------------
# CONFIG
# ---------------------------
COMPANY_FILE = "data/Company_S_HQ_1.xlsx"
SYMBOL_FILE = "data/SP500_Symbols_2.csv"
TABLE_NAME = "embeddings"

# Supabase credentials (use your .env or repo secrets)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# LOAD DATA
# ---------------------------
print("📂 Loading data...")

company_df = pd.read_excel(COMPANY_FILE)
symbols_df = pd.read_csv(SYMBOL_FILE)

print(f"✅ Loaded {len(company_df)} companies, {len(symbols_df)} symbols")

# ---------------------------
# PREP DATA
# ---------------------------
# Adjust column names if different
company_df = company_df.rename(columns={
    "Name": "name",
    "Sector": "sector",
    "Headquarters": "hq"
})

symbols_df = symbols_df.rename(columns={
    "Symbol": "symbol",
    "Security": "security"
})

# Merge on company name if available, else just keep them separate
data = []
for _, row in company_df.iterrows():
    data.append({
        "symbol": row.get("symbol", ""),
        "name": row.get("name", ""),
        "sector": row.get("sector", ""),
        "hq": row.get("hq", "")
    })

print(f"✅ Prepared {len(data)} rows for embeddings")

# ---------------------------
# EMBEDDINGS
# ---------------------------
print("🧠 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

rows_to_insert = []
for row in tqdm(data, desc="Encoding rows"):
    text = f"{row['symbol']} {row['name']} {row['sector']} {row['hq']}"
    embedding = model.encode(text).tolist()
    rows_to_insert.append({
        "symbol": row["symbol"],
        "name": row["name"],
        "sector": row["sector"],
        "hq": row["hq"],
        "embedding": embedding
    })

print(f"✅ Generated embeddings for {len(rows_to_insert)} rows")

# ---------------------------
# SAVE TO SUPABASE
# ---------------------------
print(f"⬆️ Inserting into Supabase table: {TABLE_NAME}")

# Ensure table exists: you must create it once in Supabase SQL Editor:
#   create table if not exists embeddings (
#       id bigserial primary key,
#       symbol text,
#       name text,
#       sector text,
#       hq text,
#       embedding vector(384)  -- depends on model dimensions
#   );

batch_size = 50
for i in range(0, len(rows_to_insert), batch_size):
    batch = rows_to_insert[i:i+batch_size]
    res = supabase.table(TABLE_NAME).insert(batch).execute()
    print(f"Inserted batch {i//batch_size + 1}: {len(batch)} rows")

print("🎉 Done! All embeddings inserted.")

