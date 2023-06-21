# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:56:29 2023

@author: HP
"""


import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# loading the saved model
model = pickle.load(open('model.sav', 'rb'))

df= pd.read_csv("data.csv")

st.sidebar.header("les parametres d'entrées")
st.sidebar.write('''
# Application prédiction de crise cardique
Cette Application prédite si le patient aura une crise cardiaque lors des analyses  
Auteur: Parfait Tanoh N'goran
''')


st.title("Application de prédiction :ship:")
ID = st.text_input("Entrer le numero de la deamnde") 
Demande_P1 = st.text_input("Entrer la demande du prdruit")

def predict():
    data = {'ID': int(ID), 'Demande_P1': float(Demande_P1)}

    X = pd.DataFrame(data,index=[0])
    prediction = model.predict(X) 

trigger = st.button('Predict', on_click=predict)
