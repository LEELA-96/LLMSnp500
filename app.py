import streamlit as st
from supabase import create_client
import pandas as pd

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("S&P 500 Stock Data Viewer")

# Fetch data
data = supabase.table('stock_data').select('*').execute().data
df = pd.DataFrame(data)
st.dataframe(df)

query = st.text_input("Enter stock symbol to view metadata:")
if query:
    meta = supabase.table('company_metadata').select('*').eq('symbol', query.upper()).execute().data
    st.write(meta)

