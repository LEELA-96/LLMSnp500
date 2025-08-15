"""
LLMSnp500 - Stock Data + Embeddings Pipeline
Author: Beginner-friendly guide
Description:
    1. Reads S&P 500 symbols and company metadata from GitHub data folder
    2. Fetches 5 years of historical stock data (incremental updates included)
    3. Generates embeddings for stock data
    4. Inserts data and embeddings into Supabase PostgreSQL
"""

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
from supabase import create_client
from sentence_transformers import SentenceTransformer
import datetime
from tqdm import tqdm

# ---------------- Supabase Configuration ----------------
SUPABASE_URL = "https://bhssymcperznabnnzyin.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJoc3N5bWNwZXZibm56eWluIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyNzA0MzIsImV4cCI6MjA3MDg0NjQzMn0.iItw3XrlZ1Flg_s1zmtfff6uzOTkhDpxnhxyXBLuB5Q"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
engine = create_engine(f'postgresql://postgres:{SUPABASE_KEY}@bhssymcperznabnnzyin.supabase.co:5432/postgres')

# ---------------- Load Data ----------------
symbols_df = pd.read_csv('data/SP500_Symbols (2).csv')
hq_df = pd.read_excel('data/Company_S&HQS (1).xlsx')

# Merge company metadata
companies = pd.merge(symbols_df, hq_df, how='left', on='Symbol')

# ---------------- Determine Last Update ----------------
query = "SELECT MAX(date) as last_date FROM stock_data"
try:
    last_date = pd.read_sql(query, engine)['last_date'][0]
except:
    last_date = None

if last_date is None:
    start_date = datetime.datetime.today() - datetime.timedelta(days=5*365)  # 5 years initial
else:
    start_date = last_date + datetime.timedelta(days=1)  # next day after last update

end_date = datetime.datetime.today()

print(f"Fetching stock data from {start_date.date()} to {end_date.date()}...")

# ---------------- Fetch Stock Data ----------------
all_stock_data = []

for symbol in tqdm(companies['Symbol'], desc="Downloading Stocks"):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if not df.empty:
            df['Symbol'] = symbol
            df.reset_index(inplace=True)
            all_stock_data.append(df)
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

if not all_stock_data:
    print("No new data to fetch.")
    exit()

stock_data_df = pd.concat(all_stock_data, ignore_index=True)

# ---------------- Insert Stock Data ----------------
stock_data_df.to_sql('stock_data', engine, if_exists='append', index=False)
print(f"Inserted {len(stock_data_df)} rows into stock_data table.")

# ---------------- Insert Company Metadata ----------------
for _, row in companies.iterrows():
    supabase.table("company_metadata").upsert({
        "symbol": row["Symbol"],
        "name": row.get("Name", ""),
        "headquarters": row.get("Headquarters", ""),
        "previous_headquarters": row.get("Previous_Headquarters", None)
    }).execute()

print("Company metadata upsert complete.")

# ---------------- Generate Embeddings ----------------
model = SentenceTransformer('all-MiniLM-L6-v2')

for _, row in tqdm(stock_data_df.iterrows(), total=len(stock_data_df), desc="Generating Embeddings"):
    text = f"{row['Symbol']} {row['Date']} close {row['Close']}"
    embedding = model.encode(text).tolist()
    supabase.table("stock_embeddings").upsert({
        "symbol": row['Symbol'],
        "date": str(row['Date']),
        "embedding": embedding
    }).execute()

print("Embeddings generation and storage complete.")

