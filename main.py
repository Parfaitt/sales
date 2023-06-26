import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Loading the saved models
ventes1 = pickle.load(open('ventes.pkl', 'rb'))
ventes2 = pickle.load(open('ventes2.pkl', 'rb'))
ventes3 = pickle.load(open('ventes3.pkl', 'rb'))
ventes4 = pickle.load(open('ventes3.pkl', 'rb'))

# Creating a function for Prediction
def pred(ventes, ID, Demande):
    input_data = np.array([[Demande]], dtype='float')
    prediction = ventes.predict(input_data)
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
    
    Auteur: Mostapha ASLOUJ
    ''')
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Jeux de données brutes")
        st.write(df_sample)
    
    # Giving a title
    st.title("Application Web de prédiction de la vente en fonction de la demande")
    
    # Getting input from the user
    ID = st.number_input('Le numéro du produit', min_value=1, max_value=4, step=1)
    Demande = st.text_input('La valeur de la demande')
    
    # Code for prediction
    diagnosis = ''
    if st.button("Prédire"):
        if pd.notna(ID) and pd.notna(Demande):
            if ID == 1:
                diagnosis = pred(ventes1, ID, Demande)
            elif ID == 2:
                diagnosis = pred(ventes2, ID, Demande)
            elif ID == 3:
                diagnosis = pred(ventes3, ID, Demande)
            elif ID == 4:
                diagnosis = pred(ventes1, ID, Demande)
            st.success(f"La prédiction pour le produit {ID} avec une demande de {Demande} est : {diagnosis}")
        else:
            st.error("Veuillez fournir une valeur valide pour le numéro du produit et la demande.")

    # Getting the input data from the user

if __name__ == '__main__':
    main()
