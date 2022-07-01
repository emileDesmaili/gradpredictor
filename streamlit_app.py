import streamlit as st
import pandas as pd

st.header("gradpredictor")
st.subheader("Hello World")
st.write("Predicting graduate admissions")

data = pd.read_csv('data/admission_data.csv')

