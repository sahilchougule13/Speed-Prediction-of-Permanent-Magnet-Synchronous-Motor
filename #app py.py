# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 19:51:16 2023

@author: Sahil
"""


import numpy as np
import pandas as pd
import streamlit as st
import pickle 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression,RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Predicting by uploading the file")

data=pd.read_csv("temperature_data.csv")
#model=pickle.load(open('Regression.pkl','rb'),encoding='UTF-8')
st.header("PMSM speed Prediction")

def read_file():
    uploaded=st.file_uploader("upload file",type={'csv','txt'}) 
    if uploaded is not None:
        df=pd.read_csv(uploaded)
        st.write(df)
        return df

x=pd.DataFrame(read_file())
st.sidebar.subheader("Select the Model")
algo=st.sidebar.selectbox("Model: ",["Linear Regression","Robust Regression","Polynomial Regression","KNN","Decision Tree","Random Forest","Neural Networks"])


if (algo == 'Linear Regression'):
    model=LinearRegression()
elif(algo == 'Robust Regression'):
    model = RANSACRegressor(base_estimator=LinearRegression())
elif(algo == "Polynomial Regression"):
     model=PolynomialFeatures(degree = 2)
elif(algo == "KNN"):
    model=KNeighborsRegressor()
elif(algo == "Decision Tree"):
    model=DecisionTreeRegressor()
elif(algo == "Random Forest"):
    model=RandomForestRegressor()
elif(algo == "Neural Networks"):
    model=Sequential()
    model.add(Dense(10,input_dim=2,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='Adam')
    

def predict_fun():
    X=data[['u_q','i_d']]
    Y=data['motor_speed']
    model.fit(X,Y)    
    prediction=model.predict(X)
    data['Predicted_Motorspeed']=prediction
    return data
    
if st.button('PREDICT'):
    x=predict_fun()
    st.write(x[['motor_speed','Predicted_Motorspeed']])
    fig=plt.figure(figsize=(6,4))
    sns.distplot(x['motor_speed'],hist=False,color='green')
    sns.distplot(x['Predicted_Motorspeed'],hist=False,color='red')
    plt.legend(['Actual speed','Predicted speed'])
    plt.title("Actual speed v/s Predicted speed",fontdict={'color':'blue','size':20})
    st.pyplot(fig)

df=predict_fun()    
 
if st.button('Download File'):
        df.to_csv('Predict_pmsmspeed.csv')
