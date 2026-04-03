import streamlit as st
import pandas as pd
from data_cleaning import load_data, clean_data
from risk_engine import apply_risk_logic

df = load_data("data/dirtyFile.csv")
df_clean = clean_data(df)
df_final = apply_risk_logic(df_clean)

st.title("Delivery Risk Dashboard")
st.dataframe(df_final[['job_id','driver_id','delay','risk_score','risk_level','recommended_action']])
st.bar_chart(df_final['risk_level'].value_counts())