import streamlit as st
import pandas as pd
from pandas import read_csv
from datetime import datetime
import matplotlib.pyplot as plt

# load data
dataset = pd.read_csv('pas_001_all_feature.csv',parse_dates=[['date','time']],index_col=[0])

st.write(dataset)

st.subheader("DataFrame Cleaned, Date and Time index")
    #เรียงลำดับตามวันเวลา
df = dataset.sort_index(level="date")
    
    # ลบแถวที่ไม่มีข้อมูลออก
exclude_data = df[ (df['water_level']=='-') | (df['rain_hourly']=='-') | (df['temperature']=='-') | (df['humidity']=='-') | (df['pressure']=='-') 
                  | (df['water_level']=='-999') | (df['rain_hourly']=='-999') | (df['temperature']=='-999') | (df['humidity']=='-999') | (df['pressure']=='-999') 
                    ].index
df.drop(exclude_data,inplace=True)
st.write('DataFrame shape after cleaning, (exclude "-" and "-999")', df.shape)
st.write('Cleaned and Sort index in DataFrame by Date & Time', df)

st.subheader("DataFrame convert data type to Float")
df_to_train = df.astype(float)
st.write('DataFrame shape', df_to_train.shape)

st.write("Data Frame head 10 rows",df_to_train.head(10))

df.to_csv('pas_001_clean.csv')

dataset = read_csv('pas_001_clean.csv', header=0, index_col=0)
values = dataset.values
st.write(dataset)
st.write(values)
groups = [0,1,2,3,4]
i = 1
plt.figure()
for group in groups:
   plt.subplot(len(groups), 1, i)
   plt.plot(values[:, group])
   plt.title(dataset.columns[group], y=0.5, loc='right')
   i += 1
#plt.show()
st.pyplot(plt)
