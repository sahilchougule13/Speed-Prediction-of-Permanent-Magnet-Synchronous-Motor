# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 18:43:38 2023

@author: Chandrika
"""


import numpy as np
import pandas as pd
import streamlit as st
from sklearn import linear_model
from sklearn.linear_model import LinearRegression,RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense

st.set_page_config(page_title="Predicting for single input")

data=pd.read_csv("temperature_data.csv")
st.header("PMSM speed Prediction")
st.sidebar.header("User Input Parameter")

def user_input_prediction():
    uq=st.sidebar.slider("Enter Voltage q-component value",-3.00,3.00,0.01)
    id=st.sidebar.slider("Enter Current i-component value",-3.00,3.00,0.01)
    algo=st.sidebar.selectbox("Model: ",["KNN","Linear Regression","Robust Regression","Decision Tree","Random Forest","Neural Networks"])
    data={'u_q':[uq],'i_d':[id],'algo':[algo]}
    df=pd.DataFrame(data)
    return df
 
x=user_input_prediction()
algo=x.iloc[:,-1]
x=x.drop('algo',axis=1)
st.write(x)
st.write("Predicting using the model:",algo)

if (algo == 'KNN').all():
    model=KNeighborsRegressor()
elif(algo == 'Robust Regression').all():
    model = RANSACRegressor(base_estimator=LinearRegression())
elif(algo == "Linear Regression").all():
    model=LinearRegression()
elif(algo == "Decision Tree").all():
    model=DecisionTreeRegressor()
elif(algo == "Random Forest").all():
    model=RandomForestRegressor()
elif(algo == "Neural Networks").all():
    model=Sequential()
    model.add(Dense(10,input_dim=2,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='Adam')
    
X=data[['u_q','i_d']]
Y=data['motor_speed']
model.fit(X,Y)
        
st.subheader('Predict Result')

if st.button('PREDICT'):
    prediction=model.predict(x)
    st.write("The predicted motor speed is:",prediction)