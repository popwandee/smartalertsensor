import streamlit as st
import numpy as np  # woriking with array
import pandas as pd # data processiong

st.header("Data Analytics and Machine Learning Page")
st.subheader("Original data from source")
df = pd.read_csv('pas_001_all_feature.csv',index_col=['date','time'])

st.write('DataFrame shape', df.shape)
st.write('Columns in DataFrame', df.columns)
st.write('Data types in DataFrame', df.dtypes)
st.write('Index in DataFrame', df.index)
st.write('DataFrame',df)