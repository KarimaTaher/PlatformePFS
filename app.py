from flask import Flask, render_template, jsonify , request
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.losses import mse
from datetime import datetime
import pandas as pd
import json
from model2.predict_xgb import predict_with_xgb

app = Flask(__name__)

# Charger le modèle sans custom_objects pour l'instant
model = tf.keras.models.load_model('model/lstm_model.h5', custom_objects={'mse': mse})

# Charger les autres objets (scaler, target_index, scaled_data)
scaler = joblib.load('model/scaler.save')
target_index = joblib.load('model/target_index.save')
scaled_data = joblib.load('model/scaled_data.save')


def get_price_evolution():
   df = pd.read_excel('model/petrole_ai.xlsx',header=1)
   df.columns = ['Date','Prix', 'Import (Thousand Barrels )', 'Export',
               'Production(Thousand Barrels per Day)', 'Inflation (%)',
               'GDP(Billions of USD)', 'Event']
   df = df.dropna(subset=['Date', 'Prix'])
   df['Date'] = pd.to_datetime(df['Date'])
   data = {
      "labels": df['Date'].dt.strftime('%Y-%m').tolist(),
      "prices": df['Prix'].tolist()
   }
   return data
def get_full_data():
   df = pd.read_excel('model/petrole_ai.xlsx', header=1)
   df.columns = ['Date','Prix', 'Import (Thousand Barrels )', 'Export',
               'Production(Thousand Barrels per Day)', 'Inflation (%)',
               'GDP(Billions of USD)', 'Event']

   df = df.dropna(subset=['Date', 'Prix'])
   df['Date'] = pd.to_datetime(df['Date'])

   # Simplification pour envoyer au front
   # Convertir tout en dict JSON compatible (dates en string ISO)
   data_dict = df.to_dict(orient='list')
   data_dict['Date'] = [d.strftime('%Y-%m-%d') for d in df['Date']]
   return df, data_dict

def get_scatter_data(df):
   
   # On retire les lignes sans inflation ou prix
   df_filtered = df.dropna(subset=['Inflation (%)', 'Prix'])
   data = [{"x": round(row['Inflation (%)'], 2), "y": round(row['Prix'], 2)} for _, row in df_filtered.iterrows()]
   return data

def get_correlation_matrix(df):
   vars = ['Prix', 'Inflation (%)', 'Production(Thousand Barrels per Day)', 'Import (Thousand Barrels )']
   corr_df = df[vars].corr()
   variables = vars
   corr_matrix = corr_df.values.tolist()
   return variables, corr_matrix

from scipy.stats import zscore

def get_multi_time_series(df):
   df_sorted = df.sort_values('Date').dropna(subset=['Production(Thousand Barrels per Day)', 'Import (Thousand Barrels )', 'Export'])
   labels = df_sorted['Date'].dt.strftime('%Y-%m').tolist()
   production = zscore(df_sorted['Production(Thousand Barrels per Day)']).round(2).tolist()
   imp = zscore(df_sorted['Import (Thousand Barrels )']).round(2).tolist()
   export = zscore(df_sorted['Export']).round(2).tolist()
   return {
      "labels": labels,
      "production": production,
      "import": imp,
      "export": export
   }
def get_boxplot_data(df):
   # Garder seulement lignes avec prix et event non null
   df_box = df.dropna(subset=['Prix', 'Event'])
   
   categories = [
      "Global Economic Events",
      "Oil Production & Policy",
      "Geopolitical Events",
      "Natural Disasters"
   ]
   min_vals, q1_vals, med_vals, q3_vals, max_vals = [], [], [], [], []

   for cat in categories:
      group = df_box[df_box['Event'] == cat]['Prix']
      if len(group) == 0:
            min_vals.append(0)
            q1_vals.append(0)
            med_vals.append(0)
            q3_vals.append(0)
            max_vals.append(0)
      else:
            min_vals.append(group.min())
            q1_vals.append(group.quantile(0.25))
            med_vals.append(group.median())
            q3_vals.append(group.quantile(0.75))
            max_vals.append(group.max())

      return {
      "labels": categories,
      "datasets": [
            {"label": "Min", "data": min_vals},
            {"label": "Q1", "data": q1_vals},
            {"label": "Médiane", "data": med_vals},
            {"label": "Q3", "data": q3_vals},
            {"label": "Max", "data": max_vals},
      ]
   }
   
def get_event_pie_data(df):
   counts = df['Event'].value_counts()
   labels = counts.index.tolist()
   data = counts.tolist()
   return {
      "labels": labels,
      "data": data
   }
   
def get_seasonality_data(df):
   df_sorted = df.sort_values('Date').dropna(subset=['Prix'])
   labels = df_sorted['Date'].dt.strftime('%Y-%m').tolist()
   prix = df_sorted['Prix'].tolist()
   # Tendance = moyenne mobile 6 mois
   tendance = prix.copy()
   for i in range(len(prix)):
      window = prix[max(0,i-2):min(len(prix), i+3)]  # fenêtre centrée 5 points
      tendance[i] = sum(window)/len(window)
   saison = [round(p - t, 2) for p, t in zip(prix, tendance)]
   return {
      "labels": labels,
      "prix": prix,
      "tendance": [round(x, 2) for x in tendance],
      "saison": saison
   }



@app.route('/predictions', methods=['GET', 'POST'])
def prediction():
   global model
   selected_model = request.args.get("model", "LSTM")  # LSTM par défaut
   if selected_model == "XGBoost":
      # Chargement du modèle XGBoost et de l'encoder
      model = joblib.load("model2/xgb_model.pkl")
      encoder = joblib.load("model2/xgb_encoder.pkl")

      # Chargement des données de test
      X_test = pd.read_csv("model2/x_test.csv")

      # Appliquer l'encodage si nécessaire
      # (seulement si tu n'as pas déjà encodé X_test avant de le sauvegarder)
      try:
         X_test_encoded = encoder.transform(X_test)
      except:
         X_test_encoded = X_test  # Si déjà encodé

      # Prendre la dernière ligne pour la prédiction
      last_row = X_test_encoded[-1:]
      prediction = model.predict(last_row)[0]
      prix_arrondi = round(prediction, 2)

      # Informations du modèle
      model_info = {
         "nom": "XGBoost",
         "type": "Boosting",
         "date": "2025-05-17",
         "precision": "91%"
      }
   else:
      # LSTM par défaut
      last_input = scaled_data[-24:]
      next_input = np.expand_dims(last_input, axis=0)
      next_pred_scaled = model.predict(next_input)
      next_pred = scaler.inverse_transform(np.concatenate([
            np.zeros((1, target_index)),
            next_pred_scaled,
            np.zeros((1, scaled_data.shape[1] - target_index - 1))
      ], axis=1))[:, target_index][0]
      prix_arrondi = round(next_pred, 2)
      model_info = {
            "nom": "LSTM",
            "type": "Réseau de neurones",
            "date": "2025-05-17",
            "precision": "92%"
      }
      
   date_maj = datetime.now().strftime("%d %B %Y")
   evolution_data = get_price_evolution()
   df, _ = get_full_data()  
   
   # Visualisations
   scatter_inflation_data = get_scatter_data(df)
   variables_names, corr_matrix = get_correlation_matrix(df)
   multi_time_series = get_multi_time_series(df)
   boxplot_data = get_boxplot_data(df)
   seasonality_data = get_seasonality_data(df)
   event_pie_data = get_event_pie_data(df)

   return render_template('prediction.html', prix=prix_arrondi, date_maj=date_maj, evolution_data=evolution_data, full_data=df,    
                           scatter_inflation_data=scatter_inflation_data,
                           variables_names=variables_names,
                           corr_matrix=corr_matrix,
                           multi_time_series=multi_time_series,
                           boxplot_data=boxplot_data,
                           seasonality_data=seasonality_data,
                           event_pie_data=event_pie_data,
                           selected_model=selected_model,
                           model_info=model_info)
   

@app.route('/')
def homepage():
   return render_template('index.html')

if __name__ == '__main__':
   app.run(debug=True, host='127.0.0.1', port=5001)
