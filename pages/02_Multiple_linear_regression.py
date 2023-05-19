import streamlit as st
import numpy as np  # woriking with array
import pandas as pd # data processiong
import folium
from streamlit_folium import st_folium, folium_static
import matplotlib.pyplot as plt # visualizations
import plotly.express as px 
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

st.subheader('Multiple Linear Regression')
st.markdown('Station metadata')
station = pd.DataFrame({
    "code":['PAS001','PAS002','PAS003','PAS004'],
    'station name': ['หล่มสัก','เมืองเพชรบูรณ์','หนองไผ่','วิเชียรบุรี'],
    'tambon':['หล่มสัก','สะเดียง','นาเฉลียง','ท่าโรง'],
    'amphoe':['หล่มสัก','เมืองเพชรบูรณ์','หนองไผ่','วิเชียรบุรี'],      
    'province':['เพชรบูรณ์','เพชรบูรณ์','เพชรบูรณ์','เพชรบูรณ์'],        
    'lat' : [16.7814816,16.4479424,16.1149808,15.6558832],
    'lon' : [101.2467136,101.170144,101.0970816,101.100768], 
    #'basin':['แม่น้ำป่าสัก','แม่น้ำป่าสัก','แม่น้ำป่าสัก','แม่น้ำป่าสัก'],
})

st.write('Station metadata แม่น้ำป่าสักในพื้นที่จังหวัดเพชรบูรณ์: ',station)
map_data = pd.DataFrame({
   'lat':[16.7814816,16.4479424,16.1149808,15.6558832],
   'lon':[101.2467136,101.170144,101.0970816,101.100768]
   } )

st.map(map_data,zoom=6)

m = folium.Map(location=[station.lat.mean(), station.lon.mean()], 
                 zoom_start=8, control_scale=True)

#Loop through each row in the dataframe
for i,row in station.iterrows():
    #Setup the content of the popup
    iframe = folium.IFrame('สถานี:' + str(row["code"])+ str(row["station name"])+ ' ตำบล' + str(row["tambon"])+  ' อำเภอ' + str(row["amphoe"])+  ' จังหวัดเพชรบูรณ์')
    
    #Initialise the popup using the iframe
    popup = folium.Popup(iframe, min_width=300, max_width=300)
    
    #Add each row to the map
    folium.Marker(location=[row['lat'],row['lon']],
                  popup = popup, c=row['lat']).add_to(m)

st_data = folium_static(m, width=700)

# อ่านข้อมูลจากไฟล์
st.cache_data()
df = pd.read_csv('pas_001_all_feature.csv',parse_dates=[['date','time']],index_col=[0])

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

#st.write("Data Frame head 10 rows",df_to_train.head(10))

#st.write("Data Frame show all",df_to_train)

st.write("Data Frame Describe",df_to_train.describe())

st.write("Data Frame info",df_to_train.info())
dataset = df_to_train
values = dataset.values
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


feather_selection_data_by_date = df_to_train.loc['2022-01-01 01:00:00',['temperature','humidity','pressure','rain_hourly']]
water_level_data_in_date = df_to_train.loc['2022-01-01 01:00:00',['water_level']]
col1,col2 = st.columns(2)
with col1:
    st.write('Featuure selection data @ 2022-01-01 01:00:00',feather_selection_data_by_date)
with col2:
    st.write('Water level data @ 2022-01-01 01:00:00',water_level_data_in_date)

st.subheader('Split data for Independent and Dependent variable')
# Select data for modeling
X_var = df_to_train[['rain_hourly','temperature','humidity','pressure']]
y_var = df_to_train['water_level']

col1,col2 = st.columns(2)
with col1:
    st.write('X_var samples (Independent variable): ', X_var.shape, X_var)
with col2:
    st.write('y_var samples (Dependent variable): ', y_var.shape, y_var)

X_var = StandardScaler().fit(X_var).transform(X_var)

st.subheader('Split data for train and test')
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 4)
col1,col2 = st.columns(2)
with col1:
    st.write('X_train samples : ',X_train.shape, X_train)
    st.write('X_test samples : ', X_test.shape, X_test)
with col2:
    st.write('y_train samples : ', y_train.shape, y_train)
    st.write('y_test samples : ', y_test.shape, y_test)

st.header('Modelling (Multiple Linear Regression)')

model = linear_model.LinearRegression()

model.fit(X_train,y_train,sample_weight=2)

predict = model.predict(X_test)
col1,col2 = st.columns(2)
with col1:
    st.write('y_test: ',y_test.shape,y_test)
with col2:
    st.write('Predict result: ',predict.shape,predict)

st.write('Type of y_test:',type(y_test))#class series
st.write('index:',y_test.index[0])
for i in range(1,y_test.size) :
    st.write(y_test[y_test.index[i]])
    st.write(y_test.index[i])
counts=y_test.value_counts() #นับจำนวนแต่ค่าใน element
st.write('counts:',counts)
#loc        # property Access a group of rows and columns by label(s) or a boolean array.
#values()   #property Return Series as ndarray or ndarray-like depending on the dtype.
data_values=y_test.values 
st.write('values:',data_values)
#size       #property Return the number of elements in the underlying data.
data_size=y_test.size 
st.write('size:',data_size)
#name       #property Return the name of the Series.
#isin       #function Whether elements in Series are contained in `values`.
#to_list    #function Return a list of the values.
plotdata = y_test.plot()
st.write(plotdata)
st.write('Type of predict:',type(predict))#array
#line_chart = pd.DataFrame({'index' : [f"{d}" for d in list(1,6478)],'y_test': [y_test],'y_predict':[predict] })
    #columns=['Index','y_test','predict'])
#st.write(line_chart)
#st.line_chart(line_chart)

#model evaluatin
score = r2_score(y_test,predict)
st.write('r2_Score: ', score)
