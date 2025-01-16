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

app_mode = st.sidebar.selectbox('Select Page', ['Prediction','Modele'])

if app_mode == 'Modele':
   
    # Titre de l'application
    st.title("Affichage du Modèle et de ses Coefficients")

        # Charger le modèle
    with open("ElasticNet_model_fit_2.pkl", "rb") as file:
        model_pipeline = pickle.load(file)
        st.write(model_pipeline)

    # Vérifier si le modèle contient une étape de régression linéaire
    try:
        # Accès au modèle final (dans le pipeline)
        linear_model = model_pipeline.named_steps['model']  # Adapter en fonction du nom de l'étape du modèle
    except AttributeError:
        st.error("Impossible de trouver l'étape de régression dans le pipeline. Vérifiez le fichier chargé.")
        linear_model = None

if app_mode == 'Prediction':    
    # Charger le modèle
    with open('ElasticNet_model_fit_2.pkl', 'rb') as file:
        model = pickle.load(file)

        #st.write("Étapes du pipeline :", model.named_steps)

        st.title("Prédiction avec un Modèle de Régression Linéaire ElasticNet")
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
        sex_mapping = {"male": "male", "female":"female"}
        smoker_mapping = {"yes": "yes", "no": "no"}
        region_mapping = {"southeast":"southeast", "northwest":"northwest", "northeast":"northeast", "southwest":"southwest"}

        sex = sex_mapping[sex]
        smoker = smoker_mapping[smoker]
        region = region_mapping[region]

        bmi_smoker = bmi * (1 if smoker == "yes" else 0)
        age_smoker = age * (1 if smoker == "yes" else 0)
        age_bmi = age * bmi

        bins_age = [0, 28, 51, 65, np.inf]
        labels_age = ['Jeune', 'Mature', 'Âgé', 'Senior']
        
        bins_bmi = [0, 18, 30, 40, np.inf]
        labels_bmi = ['Maigre', 'Normal', 'Surpoids', 'Obèse']
        
        # Trouver le groupe d'âge correspondant
        age_group = labels_age[np.digitize(age, bins_age, right=False) - 1]
        bmi_category = labels_bmi[np.digitize(bmi, bins_bmi, right=False) - 1]

        input_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "age_group": age_group,
            "bmi_category": bmi_category,
            "bmi_smoker": bmi_smoker,
            "age_smoker": age_smoker,
            "age_bmi": age_bmi
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