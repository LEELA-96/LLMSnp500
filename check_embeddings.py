import os
from supabase import create_client

# -----------------------------
# 1️⃣ Supabase setup
# -----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# 2️⃣ Fetch embeddings
# -----------------------------
res = supabase.table("stock_embeddings").select("*").limit(5).execute()
data = res.data

if not data:
    print("❌ No embeddings found! Run update_data.py first.")
else:
    print(f"✅ Found {len(data)} embeddings. Sample rows:")
    for row in data:
        print(f"Symbol: {row['symbol']}, Date: {row['date']}, Embedding length: {len(row['embedding'])}")
