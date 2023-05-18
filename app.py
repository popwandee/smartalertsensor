import streamlit as st
import numpy as np  # woriking with array
import pandas as pd # data processiong
import itertools    # construct specialized tools
import matplotlib.pyplot as plt # visualizations
import time 
from matplotlib import rcParams # plot size customization
from termcolor import colored as cl # text customization
from sklearn.model_selection import train_test_split # splitting the data
from sklearn import linear_model # Model algorithm
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.metrics import mean_squared_error as jss #evaluation metric
from sklearn.metrics import r2_score  #evaluation metric
from sklearn.metrics import precision_score # evaluation metric
from sklearn.metrics import classification_report # evaluation metric
from sklearn.metrics import confusion_matrix #evaluation metric
from sklearn.metrics import log_loss #evaluation metric

st.set_page_config(
    page_title="Data Analytics and Machine Learning Page",
    page_icon="👋",
)
rcParams['figure.figsize'] = (20,10)
st.header("Data Analytics and Machine Learning Page")

st.subheader('Main page')
# อ่านข้อมูลจากไฟล์
st.cache_data()
df = pd.read_csv('pas_001_all_feature.csv',index_col=['date','time'])

st.subheader("DataFrame from source")
st.write('DataFrame shape', df.shape)
st.write('Columns in DataFrame', df.columns)
st.write('Data types in DataFrame', df.dtypes)
st.write('Index in DataFrame', df.index)
st.write('DataFrame',df)

st.subheader("DataFrame Cleaned, Date and Time index")
    #เรียงลำดับตามวันเวลา
df = df.sort_index(level="date")
    
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

#st.write("Data Frame show all",df_to_train)

st.write("Data Frame Describe",df_to_train.describe())

st.write("Data Frame info",df_to_train.info())

feathur_selection_data_by_date = df_to_train.loc['2022-01-01',['temperature','humidity','pressure','rain_hourly']]
water_level_data_in_date = df_to_train.loc['2022-01-01',['water_level']]
col1,col2 = st.columns(2)
with col1:
    st.write('Featuure selection data by date',feathur_selection_data_by_date)
with col2:
    st.write('Water level data by date',water_level_data_in_date)

# Select data for modeling
X_var = df_to_train[['rain_hourly','temperature','humidity','pressure']]
y_var = df_to_train['water_level']

col1,col2 = st.columns(2)
with col1:
    st.write('X_var samples: ', X_var.shape, X_var)
with col2:
    st.write('y_var samples: ', y_var.shape, y_var)

X_var = StandardScaler().fit(X_var).transform(X_var)

st.subheader('Split data for train and test')
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 4)
col1,col2 = st.columns(2)
with col1:
    st.write('X_train samples : ',X_train.shape, X_train)
    st.write('X_test samples : ', X_test.shape, X_test)
with col2:
    st.write('y_train samples : ', y_train.shape, y_train)
    st.write('y_test samples : ', y_test.shape, y_test)

st.header('Modelling (Logistic Regression with scikit-learn)')

model = linear_model.LinearRegression()

model.fit(X_train,y_train)

predict = model.predict(X_test)
st.write('Predict result: ',predict.shape,predict)

#model evaluatin
score = r2_score(y_test,predict)
st.write('Score: ', score)