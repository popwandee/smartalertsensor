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

st.subheader('Original data Source')
st.subheader('Multiple linear regression')
st.subheader('Multivariate Time Series with LSTM ')
