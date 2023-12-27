from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = joblib.load('Modèle de prédiction CatBoostClassifier')


# Fonction pour effectuer la prédiction
def prediction(data):
    # Extrayez les données du JSON
    amt_credit = data['amt_credit']
    amt_annuity = data['amt_annuity']
    days_birth = data['days_birth']
    days_employed = data['days_employed']
    days_registration = data['days_registration']
    days_id_publish = data['days_id_publish']
    ext_source_2 = data['ext_source_2']
    ext_source_3 = data['ext_source_3']
    days_last_phone = data['days_last_phone']
    amount_previous_credit = data['amount_previous_credit']


    new_data = np.array([amt_credit, amt_annuity, days_birth, days_employed, days_registration, days_id_publish,
                         ext_source_2, ext_source_3, days_last_phone, amount_previous_credit])
    pred = model.predict(new_data.reshape(1, -1))
    if pred == 0:
        return "Crédit accordé"
    else:
        return "Crédit refusé"


# Routes Flask
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    result = prediction(data)
    return jsonify({'prediction': result})

# Nouvelle route pour la racine
@app.route('/')
def home():
    return 'Bienvenue sur l\'API de prédiction de crédit.'



if __name__ == '__main__':
    app.run(port=8080)




