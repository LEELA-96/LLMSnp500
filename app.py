import os
import streamlit as st
import pandas as pd
from supabase import create_client
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.express as px

# -----------------------------
# 1️⃣ Supabase setup
# -----------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# 2️⃣ Embedding model
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# 3️⃣ Streamlit UI
# -----------------------------
st.title("📊 S&P 500 LLM Query & Stock Charts")
st.write("Ask a question about any S&P 500 stock, and see the top 5 relevant matches with historical stock trends.")

query = st.text_input("Enter your question:")

if query:
    # -----------------------------
    # 4️⃣ Fetch all embeddings from Supabase
    # -----------------------------
    res = supabase.table("stock_embeddings").select("*").execute()
    embeddings_data = res.data

    if not embeddings_data:
        st.warning("No embeddings found in Supabase. Run update_data.py first.")
    else:
        symbols = [row['symbol'] for row in embeddings_data]
        dates = [row['date'] for row in embeddings_data]
        embeddings = [np.array(row['embedding']) for row in embeddings_data]

        # -----------------------------
        # 5️⃣ Compute query embedding
        # -----------------------------
        query_embedding = model.encode(query)

        # -----------------------------
        # 6️⃣ Cosine similarity
        # -----------------------------
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:5]  # Top 5 matches

        for rank, idx in enumerate(top_indices, start=1):
            best_symbol = symbols[idx]
            best_date = dates[idx]

            # -----------------------------
            # 7️⃣ Fetch stock data for this symbol
            # -----------------------------
            stock_res = supabase.table("stock_data").select("*") \
                .eq("symbol", best_symbol).execute()
            if stock_res.data:
                stock_df = pd.DataFrame(stock_res.data)
                stock_df['date'] = pd.to_datetime(stock_df['date'])

                st.subheader(f"Top {rank}: {best_symbol}")
                st.write(f"Most relevant date: {best_date}")
                st.write(stock_df.tail(5))  # Show last 5 rows as preview

                # -----------------------------
                # 8️⃣ Plot historical stock trends
                # -----------------------------
                fig = px.line(stock_df, x='date', y='close', title=f"{best_symbol} Closing Prices")
                st.plotly_chart(fig)
            else:
                st.info(f"No stock data found for {best_symbol}")
