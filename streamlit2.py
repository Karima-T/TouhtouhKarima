import streamlit as st
from flask import Flask, request, jsonify
import requests
import joblib
import shap
import pandas as pd
from lime import lime_tabular
import numpy as np
import matplotlib.pyplot as plt




# Create three columns
col1, col2, col3 = st.columns(3)

# Column 2
with col2:
    st.image("logo.png")
    
# Title and text
st.title("Scoring Bancaire")
st.markdown(
    "Cette application a pour but de prédire si un client risque d'être en défaut de paiement ou non. "
    "C'est donc une aide au conseiller bancaire pour décider si on accepte ou non d'accorder un crédit à ce client."
)

# Chargement du modèle sélectionné

model=joblib.load(filename="Modèle de prédiction CatBoostClassifier")

# Fonction pour effectuer la prédiction

def prediction(amt_credit,amt_annuity,days_birth,days_employed,days_registration,days_id_publish,
               ext_source_2,ext_source_3,days_last_phone,amount_previous_credit):
    ext_source_2=0.46
    ext_source_3=0.46
    new_data=np.array([amt_credit,amt_annuity,days_birth,days_employed,days_registration,days_id_publish,
                       ext_source_2,ext_source_3,days_last_phone,amount_previous_credit])
    pred=model.predict(new_data.reshape(1,-1))
    return pred


# Sidebar pour entrer les caractéristiques du client
st.sidebar.header("Informations client")
st.sidebar.write("Veuillez compléter les informations ci-dessous pour obtenir votre prédiction")
amt_credit=st.sidebar.number_input(label="Montant du crédit",min_value=45000,max_value=5000000,value=600000)
amt_annuity=st.sidebar.number_input(label="Montant annuel des intérêts",min_value=2500,max_value=200000,value=30000)
days_birth=st.sidebar.number_input(label="Age du client en jours",min_value=-29200,max_value=-6570,value=-15000)
days_employed=st.sidebar.number_input(label="Nombre de jours depuis le début de l'emploi en cours au moment du crédit",min_value=-18000,max_value=0,value=-10000)
days_registration=st.sidebar.number_input(label="Ancienneté du client dans la banque en jours au moment du crédit",min_value=-25000,max_value=0,value=-10000)
days_id_publish=st.sidebar.number_input(label="Ancienneté de la pièce d'identité en jours au moment du crédit",min_value=-8000,max_value=0,value=-5000)
days_last_phone=st.sidebar.number_input(label="Nombre de jours depuis le dernier achat d'un téléphone au moment du crédit",min_value=-5000,max_value=0,value=-1000)
amount_previous_credit=st.sidebar.number_input(label="Montant d'un précédent crédit",min_value=0,max_value=9000000,value=500000)
ext_source_2=0.46
ext_source_3=0.46




# Création du bouton Predict

if st.button("Predict"):
    input_data = {
        "amt_credit": amt_credit,
        "amt_annuity": amt_annuity,
        "days_birth":days_birth,
        "days_employed":days_employed,
        "days_registration":days_registration,
        "days_id_publish":days_id_publish,
        "ext_source_2":ext_source_2,
        "ext_source_3":ext_source_3,
        "days_last_phone":days_last_phone,
        "amount_previous_credit":amount_previous_credit}


    # Faire la prédiction en utilisant l'API Flask
    response = requests.post("https://api-prediction1981-00dbbf4d2fd2.herokuapp.com/predict", json=input_data)

    if response.status_code == 200:
        prediction_result = response.json()['prediction']

        if prediction_result == "Crédit accordé":
            st.success("Crédit accordé")
        else:
            st.error("Crédit refusé")
    else:
        st.error(f"Erreur de prédiction. Code d'erreur : {response.status_code}")

# ...


# Affichage des shap values:
    new_data=np.array([amt_credit,amt_annuity,days_birth,days_employed,days_registration,days_id_publish,ext_source_2,ext_source_3,
                       days_last_phone,amount_previous_credit])


    # Definition du nom des features
    feature_names = [
    "Montant crédit",
    "Montant intérêts",
    "Age",
    "Ancienneté emploi",
    "Ancienneté client banque",
    "Ancienneté CIN",
    "External Source 2",
    "External Source 3",
    "Dernier achat téléphone",
    "Montant précédent crédit"
]

    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_values=explainer.shap_values(new_data.reshape(1,-1))
    base_value = explainer.expected_value
    force_plot = shap.force_plot(base_value, shap_values, new_data.reshape(1, -1), feature_names=feature_names,matplotlib=True)

    # Affichage du force plot avec st.pyplot
    st.header("Affichage des paramètres les plus influents sur la prédiction")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(force_plot)

    # Lime
    import lime
    from lime import lime_tabular


    # Load all clients data
    all_clients_df = pd.read_csv("Ensemble_des_clients.csv")
    all_clients_df_lime=all_clients_df.drop("prediction",axis=1)
    
    # Create a DataFrame for the client being tested
    client_input_lime = np.array([amt_credit, amt_annuity, days_birth, days_employed, days_registration, days_id_publish,
                               ext_source_2, ext_source_3, days_last_phone, amount_previous_credit])

    # Convert client_input_lime to a DataFrame with the same column names
    client_input_lime_df = pd.DataFrame([client_input_lime], columns=feature_names)


    # Création de l'explainer
    explainer = lime_tabular.LimeTabularExplainer(all_clients_df_lime.values, 
                                                feature_names=feature_names, 
                                                class_names=['0','1'], 
                                                verbose=True, 
                                                mode='classification')


    # Explain the instance
    exp = explainer.explain_instance(client_input_lime_df.values[0], model.predict_proba, num_features=10)


    # Get Lime explanation as a matplotlib figure
    lime_fig = exp.as_pyplot_figure()

    # Ajouter des annotations pour les valeurs du client
    
    max_len = max(len(f"{col}: {client_input_lime_df[col].values[0]:.2f}") for col in feature_names)
    for i, col in enumerate(feature_names):
        value = client_input_lime_df[col].values[0]
        annotation = f"{col}: {value:.2f}"
    
        # Ajouter l'annotation pour le nom de la feature et la valeur
        plt.annotate(annotation, (0, i), textcoords="offset points", xytext=(max_len*(-3), 15), ha='left', fontsize=8, color='black')


    # Save Lime explanation as an image
    lime_img_path = "lime_explanation.png"
    lime_fig.savefig(lime_img_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(lime_fig)

    # Afficher l'explication Lime en tant qu'image dans Streamlit
    st.header("Affichage de l'explication Lime avec valeurs du client")
    st.image(lime_img_path)



    # Scatter plot for comparison
    st.subheader("Comparaison du client testé à l'ensemble des clients en fonction de la prédiction")

    # Load all clients data
    all_clients_df = pd.read_csv("Ensemble_des_clients.csv")

    # Create a DataFrame for the client being tested
    client_input = pd.DataFrame([[amt_credit, amt_annuity, days_birth, days_last_phone]],
                                columns=['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_LAST_PHONE_CHANGE'])

    # Filter clients for visualization (adjust the sample size as needed)
    sample_size = 500
    sampled_clients_df = all_clients_df.sample(sample_size)

    # Scatter plot with predictions on the y-axis and features on the x-axis
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    for i, col in enumerate(['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_LAST_PHONE_CHANGE']):
        row, col_idx = divmod(i, 2)

        # Scatter plot for credit accorded (prediction=0)
        axes[row, col_idx].scatter(sampled_clients_df[sampled_clients_df['prediction'] == 0][col],
                                sampled_clients_df[sampled_clients_df['prediction'] == 0]['prediction'],
                                alpha=0.5, label='Crédit Accordé')
        # Scatter plot for credit refused (prediction=1)
        axes[row, col_idx].scatter(sampled_clients_df[sampled_clients_df['prediction'] == 1][col],
                                sampled_clients_df[sampled_clients_df['prediction'] == 1]['prediction'],
                                alpha=0.5, label='Crédit Refusé')

        # Highlight the client being tested
        prediction_value = 0 if prediction_result == "Crédit accordé" else 1
        axes[row, col_idx].scatter(client_input[col], prediction_value, marker='X', color='red', label='Client Testé')

        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('Prediction')
        axes[row, col_idx].legend()

    # Show the scatter plot
    st.pyplot(fig)


