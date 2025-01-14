import streamlit as st
import time
import pandas as pd
import numpy as np
import pickle
import base64

@st.cache_data() 
def get_fvalue(val):    
    feature_dict = {"No": 1, "Yes": 2}
    for key, value in feature_dict.items():    
        if val == key:
            return value
        
def get_value(val, my_dict):
    for key, value in my_dict.items():    
        if val == key:
            return value

# Configuration de la page
st.set_page_config(layout="wide")

# Fonction pour convertir une image en Base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Fonction pour définir un arrière-plan pour la sidebar
def set_sidebar_background(image_path):
    encoded_image = get_base64_of_bin_file(image_path)
    sidebar_style = f"""
        <style>
            [data-testid="stSidebar"] {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-repeat: no-repeat;
                background-position: center;
            }}
        </style>
    """
    st.markdown(sidebar_style, unsafe_allow_html=True)
    # Appeler la fonction avec l'image souhaitée
set_sidebar_background("prime_assurance.png")

app_mode = st.sidebar.selectbox('Select Page', ['Modele', 'Prediction'])

if app_mode == 'Modele':
   
    # Charger le modèle
    with open("linear_regression_model_1.pkl", "rb") as file:
        model_pipeline = pickle.load(file)

    # Titre de l'application
    st.title("Affichage du Modèle et de ses Coefficients")

    # Vérifier si le modèle contient une étape de régression linéaire
    try:
        # Accès au modèle final (dans le pipeline)
        linear_model = model_pipeline.named_steps['linearregression']  # Adapter en fonction du nom de l'étape du modèle
    except AttributeError:
        st.error("Impossible de trouver l'étape de régression dans le pipeline. Vérifiez le fichier chargé.")
        linear_model = None

    # Si le modèle est un modèle de régression linéaire, afficher les coefficients
    if linear_model:

        st.header("Modèle et Coefficients")
        
        # Afficher le type du modèle
        st.write("**Modèle chargé :*****", type(linear_model).__name__)
        
        # Récupérer les coefficients
        if hasattr(linear_model, 'coef_') and hasattr(linear_model, 'intercept_'):
            coefficients = linear_model.coef_
            intercept = linear_model.intercept_
            st.write("Hello",model_pipeline.named_steps)
            # Afficher les coefficients
            st.write("### Coefficients du modèle")
            coef_df = pd.DataFrame({
                'Feature': model_pipeline.named_steps['polynomialfeatures'].get_feature_names_out(),
                'Coefficient': coefficients
            })
            st.dataframe(coef_df)

            # Afficher l'intercept
            st.write("### Intercept")
            st.write(intercept)
        else:
            st.error("Le modèle chargé n'a pas de coefficients (ce n'est peut-être pas un modèle de régression linéaire).")


if app_mode == 'Prediction':    
    # Charger le modèle
    with open('linear_regression_model_1.pkl', 'rb') as file:
        model = pickle.load(file)

        #st.write("Étapes du pipeline :", model.named_steps)

        st.title("Prédiction avec un Modèle de Régression Linéaire :")
        # Entrée utilisateur
        st.header("Entrer les caractéristiques")
        # Variables numériques
        age = st.number_input("Âge:", min_value=0, max_value=100, step=1)
        bmi = st.number_input("Indice de Masse Corporelle (BMI):", min_value=10.0, max_value=50.0, step=0.1)
        children = st.number_input("Nombre d'enfants:", min_value=0, max_value=10, step=1)

        # Variables catégoriques
        sex = st.selectbox("Sexe:", options=["male", "female"])
        smoker = st.selectbox("Fumeur:", options=["yes", "no"])

        # Variables ordinales
        region = st.selectbox(
            "Région:", 
            options=["southeast", "northwest", "northeast", "southwest"]
        )

        # Convertir les variables catégoriques et ordinales en format numérique
        # Remplacez ces mappings par ceux utilisés dans votre prétraitement
        sex_mapping = {"male": "male", "female":"female"}
        smoker_mapping = {"yes": "yes", "no": "no"}
        region_mapping = {"southeast":"southeast", "northwest":"northwest", "northeast":"northeast", "southwest":"southwest"}

        sex = sex_mapping[sex]
        smoker = smoker_mapping[smoker]
        region = region_mapping[region]

        input_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }

        # Ajouter un bouton pour effectuer une prédiction
        if st.button("Prédire"):
            input_features = pd.DataFrame([input_data])
            st.write(input_features.head())

            st.write("Données d'entrée :")
            st.write(input_features)

            # Combiner toutes les caractéristiques en un tableau
            #input_features = pd.DataFrame([age, sex, bmi, children, smoker, region])
            input_features = pd.DataFrame([input_data])
            # Faire une prédiction
            prediction = model.predict(input_features)
            
            # Afficher le résultat
            st.success(f"Prédiction : {prediction[0]:.2f}")