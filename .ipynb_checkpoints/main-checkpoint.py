# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:56:29 2023
Author: HP
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Loading the saved model
loaded_model = pickle.load(open('model.sav', 'rb'))

# Creating a function for Prediction
def pred(ID, Demande_P1):
    input_data = [ID, Demande_P1]
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return prediction

def main():
    def load_data():
        data = pd.read_csv("data.csv")
        return data
    
    df = load_data()
    df_sample = df.sample(5)
    st.sidebar.header("Informations")
    st.sidebar.write('''
    # PRÉDICTION DES VENTES 
    Il s'agit d'un projet d'apprentissage automatique de prédiction des ventes en fonction de la demande.
    
    Auteur: Hamza El Kadiri
    ''')
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Jeux de données brutes")
        st.write(df_sample)
    
    # Giving a title
    st.title("Application Web de prédiction de la vnete en fonction de la demande")
    
    # Getting input from the user
    ID = st.text_input('Le numéro du produit ')
    Demande_P1 = st.text_input('La valeur de la demande')
    
    ID = pd.to_numeric(ID, errors='coerce')
    Demande_P1 = pd.to_numeric(Demande_P1, errors='coerce')

    # Code for prediction
    diagnosis = ''
    
    # Getting the input data from the user
    if st.button("Prédire la vente de la demande :"):
        diagnosis = pred(ID, Demande_P1)
        st.success(diagnosis)

if __name__ == '__main__':
    main()
