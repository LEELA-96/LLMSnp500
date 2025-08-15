import os
from supabase import create_client
import pprint

# -----------------------------
# 1️⃣ Supabase setup
# -----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

pp = pprint.PrettyPrinter(indent=2)

tables = ["company_metadata", "stock_data", "stock_embeddings"]

for table in tables:
    try:
        # Count total rows
        count_res = supabase.table(table).select("id", count="exact").execute()
        total_rows = count_res.count
        print(f"✅ Table '{table}' total rows: {total_rows}")

        # Preview first 5 rows
        preview_res = supabase.table(table).select("*").limit(5).execute()
        print(f"First 5 rows of '{table}':")
        pp.pprint(preview_res.data)
        print("-" * 50)

    except Exception as e:
        print(f"❌ Error fetching table '{table}': {e}")
