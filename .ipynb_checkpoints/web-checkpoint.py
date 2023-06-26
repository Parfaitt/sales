import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Loading the saved models
model1 = pickle.load(open('model.sav', 'rb'))
model2 = pickle.load(open('model2.sav', 'rb'))
model3 = pickle.load(open('model3.sav', 'rb'))
model4 = pickle.load(open('model4.sav', 'rb'))

# Creating a function for Prediction
def pred(model, ID, Demande):
    input_data = np.array([[ID, Demande]], dtype='float')
    prediction = model.predict(input_data)
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
    st.title("Application Web de prédiction de la vente en fonction de la demande")
    
    # Getting input from the user
    ID = st.number_input('Le numéro du produit', min_value=1, max_value=4, step=1)
    Demande = st.text_input('La valeur de la demande')
    
    # Code for prediction
    diagnosis = ''
    if st.button("Prédire"):
        if pd.notna(ID) and pd.notna(Demande):
            if ID == 1:
                diagnosis = pred(model1, ID, Demande)
            elif ID == 2:
                diagnosis = pred(model2, ID, Demande)
            elif ID == 3:
                diagnosis = pred(model3, ID, Demande)
            elif ID == 4:
                diagnosis = pred(model4, ID, Demande)
            st.success(f"La prédiction pour le produit {ID} avec une demande de {Demande} est : {diagnosis}")
        else:
            st.error("Veuillez fournir une valeur valide pour le numéro du produit et la demande.")
    
    # Adding new rows to the dataset
    st.header("Ajouter de nouvelles données")
    new_ID = st.number_input('Le numéro du produit (nouvelle donnée)', min_value=1, max_value=4, step=1)
    new_Demande = st.text_input('La valeur de la demande (nouvelle donnée)')
    new_Vente = st.number_input('La valeur de la vente (nouvelle donnée)')
    
    if st.button("Ajouter"):
        if pd.notna(new_ID) and pd.notna(new_Demande) and pd.notna(new_Vente):
            new_data = pd.DataFrame([[new_ID, new_Demande, new_Vente]], columns=['ID', 'Demande', 'Vente'])
            df = pd.concat([df, new_data], ignore_index=True)
            st.success("Nouvelles données ajoutées avec succès.")
            st.write("Nouveau jeu de données :")
            st.write(df)
            df.to_csv("data.csv", index=False)
        else:
            st.error("Veuillez fournir des valeurs valides pour les nouvelles données.")

    # Getting the input data from the user

if __name__ == '__main__':
    main()
