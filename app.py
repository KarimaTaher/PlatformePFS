from flask import Flask, render_template, jsonify , request, send_file
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.losses import mse
from datetime import datetime
import pandas as pd
import json
import math
import io
from io import BytesIO
from fpdf import FPDF
from model2.predict_xgb import predict_with_xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os
import requests
from bs4 import BeautifulSoup
from apscheduler.schedulers.background import BackgroundScheduler
from data_updater import append_new_data_to_excel
app = Flask(__name__)

# Charger le modèle sans custom_objects pour l'instant
model = tf.keras.models.load_model('model/lstm_model.h5', custom_objects={'mse': mse})

# Charger les autres objets (scaler, target_index, scaled_data)
scaler = joblib.load('model/scaler_lstm.pkl')
target_index = joblib.load('model/target_index.pkl')
scaled_data = joblib.load('model/scaled_data.pkl')
y_test = pd.read_csv("model/y_test_lstm.csv").values  # les vraies valeurs
X_test = np.load("model/x_test_lstm.npy")  # assure-toi d’avoir ces fichiers


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
      X_test_xgb = pd.read_csv("model2/x_test.csv")
      y_test_xgb = pd.read_csv("model2/y_test.csv")


      # Appliquer l'encodage si nécessaire
      # (seulement si tu n'as pas déjà encodé X_test avant de le sauvegarder)
      try:
         X_test_encoded = encoder.transform(X_test_xgb)
      except:
         X_test_encoded = X_test_xgb  # Si déjà encodé

      # Prendre la dernière ligne pour la prédiction
      y_pred = model.predict(X_test_encoded)
      last_row = X_test_encoded[-1:]
      prediction = model.predict(last_row)[0]
      prix_arrondi = round(prediction, 2)
      
      # --- 4. Calcul des erreurs
      mae = mean_absolute_error(y_test_xgb, y_pred)
      rmse = math.sqrt(mean_squared_error(y_test_xgb, y_pred))
      r2 = r2_score(y_test_xgb, y_pred)

      
      # --- 5. Calcul de la précision dynamique
      precision_score = max(0, round(r2 * 100, 2))  # En pourcentage
      if precision_score >= 90:
         interpretation = "Très fiable : le modèle est très précis pour les prédictions actuelles."
      elif precision_score >= 75:
         interpretation = "Fiabilité moyenne : le modèle est globalement bon mais peut avoir des écarts."
      else:
         interpretation = "Peu fiable : le modèle présente des erreurs importantes."

      # Informations du modèle
      model_info = {
         "nom": "XGBoost",
         "type": "Boosting, Approche par arbres de décision",
         "description": """XGBoost (Extreme Gradient Boosting) est un algorithme d’apprentissage supervisé basé sur des arbres de décision successifs, où chaque nouvel arbre corrige les erreurs du précédent. Il est reconnu pour sa rapidité et sa précision, et constitue un excellent choix pour la prédiction du prix du pétrole en analysant les variables historiques.

                        Caractéristiques principales :

                        Optimisation des performances grâce à la parallélisation,

                        Régularisation intégrée pour éviter l’overfitting,

                        Gestion automatique des valeurs manquantes.

                        Avantages :

                        Très bon pour les ensembles de données tabulaires,

                        Facile à interpréter (importance des variables). """,
         "date": "2025-05-17",
         "precision": f"{precision_score}%",
         "commentaire_precision": interpretation,
         "mae": f"{mae:.2f}",
         "commentaire_mae": "L’erreur absolue moyenne mesure la différence moyenne entre les prédictions et les valeurs réelles.",
         "rmse": f"{rmse:.2f}",
         "commentaire_rmse": "L’erreur quadratique moyenne donne plus de poids aux grandes erreurs (sensibilité élevée aux erreurs importantes)."
      }
   else:
      # LSTM par défaut
      last_input = scaled_data[-12:]
      next_input = np.expand_dims(last_input, axis=0)
      next_pred_scaled = model.predict(next_input)
      next_pred = scaler.inverse_transform(np.concatenate([
            np.zeros((1, target_index)),
            next_pred_scaled,
            np.zeros((1, scaled_data.shape[1] - target_index - 1))
      ], axis=1))[:, target_index][0]
      prix_arrondi = round(next_pred, 2)
      
      
      # Prédictions sur X_test
      y_pred_scaled = model.predict(X_test)

      # Recalage des prédictions (inverse de la normalisation)
      y_pred = scaler.inverse_transform(
         np.concatenate([
            np.zeros((len(y_pred_scaled), target_index)),
            y_pred_scaled,
            np.zeros((len(y_pred_scaled), scaled_data.shape[1] - target_index - 1))
         ], axis=1)
      )[:, target_index]

      # Recalage des vraies valeurs
      y_true = scaler.inverse_transform(
         np.concatenate([
            np.zeros((len(y_test), target_index)),
            y_test.reshape(-1, 1),
            np.zeros((len(y_test), scaled_data.shape[1] - target_index - 1))
         ], axis=1)
      )[:, target_index]

      # Calculs d’erreurs
      mae = mean_absolute_error(y_true, y_pred)
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      precision = max(0, 100 - (mae / np.mean(y_true)) * 100)
      precision_arrondie = round(precision, 2)

      # Interprétation
      if precision >= 90:
         commentaire = "Excellente précision. Le modèle LSTM capture très bien les dynamiques du marché."
      elif precision >= 80:
         commentaire = "Bonne précision. Quelques fluctuations peuvent ne pas être bien anticipées."
      elif precision >= 70:
         commentaire = "Précision moyenne. Le modèle pourrait être amélioré avec plus de données ou un réglage fin."
      else:
         commentaire = "Précision faible. Le modèle a des difficultés à bien prédire les tendances du marché."

      
      model_info = {
            "nom": "LSTM",
            "type": "Réseau de neurones récurrents, Prédiction basée sur les séries temporelles",
            "description": """Le modèle LSTM (Long Short-Term Memory) est une architecture de réseaux de neurones conçue pour traiter des données séquentielles comme les séries temporelles. Il est particulièrement adapté à la prévision du prix du pétrole, car il est capable d’apprendre des relations à long terme dans les données, ce qui est crucial dans les phénomènes économiques évolutifs.

                        Fonctionnement interne :
                        LSTM utilise une mémoire interne régulée par trois portes :

                        Porte d’oubli : supprime les informations inutiles du passé,

                        Porte d’entrée : sélectionne les nouvelles informations à stocker,

                        Porte de sortie : détermine les informations à restituer.

                        Ce mécanisme permet d’adapter le modèle aux tendances et variations du marché pétrolier.

                        Avantages :

                        Capacité à retenir des séquences longues,

                        Très performant pour la modélisation de données non linéaires et bruyantes.

                        """,
            "date": "2025-05-17",
            "precision": f"{precision_arrondie}%",
            "mae": f"{mae:.2f}",
            "rmse": f"{rmse:.2f}",
            "commentaire": commentaire,
            "prediction": prix_arrondi
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


@app.route('/data')
def data_table():
   df, _ = get_full_data()
   df['Date'] = pd.to_datetime(df['Date'])
   df['Year'] = df['Date'].dt.year

   # Trier les années de la plus récente à la plus ancienne
   available_years = sorted(df['Year'].unique(), reverse=True)

   # Récupérer l’année sélectionnée (par défaut la plus récente)
   selected_year = int(request.args.get('year', available_years[0]))

   # Filtrer les données de l’année sélectionnée
   filtered_df = df[df['Year'] == selected_year]

   return render_template('data.html',
                           years=available_years,
                           selected_year=selected_year,
                           data=filtered_df.to_dict(orient='records'))


@app.route('/export/excel')
def export_excel():
   df, _ = get_full_data()
   output = io.BytesIO()
   with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
      df.to_excel(writer, index=False, sheet_name='Data')
   output.seek(0)
   return send_file(output, download_name="donnees_petrole.xlsx", as_attachment=True)

def clean_text_for_pdf(text):
   replacements = {
      '–': '-',   # tiret moyen
      '—': '-',   # tiret long
      '“': '"',
      '”': '"',
      '’': "'",
      '‘': "'",
      '…': '...',
      '•': '-',
      '°': ' degrees',
      '€': 'EUR',
      '©': '(c)',
      '®': '(R)',
      '™': '(TM)',
      '→': '->',
      '←': '<-',
      '±': '+/-',
      '×': 'x',
      '\u00a0': ' ',  # espace insécable
   }
   for unicode_char, ascii_equiv in replacements.items():
      text = text.replace(unicode_char, ascii_equiv)
   return text

@app.route('/export/pdf')
def export_pdf():
   df, _ = get_full_data()

   # Renommer les colonnes
   df = df.rename(columns={
      'Date': 'Date',
      'Prix': 'Price',
      'Import (Thousand Barrels )': 'Import',
      'Export': 'Export',
      'Production(Thousand Barrels per Day)': 'Production',
      'Inflation (%)': 'Inflation',
      'GDP(Billions of USD)': 'GDP',
      'Event': 'Event'
   })

   # Convertir Date en datetime et grouper par année
   df['Date'] = pd.to_datetime(df['Date'])
   df['Year'] = df['Date'].dt.year
   df_annual = df.groupby('Year')[['Price', 'Import', 'Export', 'Production', 'Inflation', 'GDP']].mean().reset_index()

   # Créer le graphique
   plt.figure(figsize=(10, 5))
   for column in ['Price', 'Import', 'Export', 'Production', 'Inflation', 'GDP']:
      plt.plot(df_annual['Year'], df_annual[column], label=column)
   plt.title('Annual Trends')
   plt.xlabel('Year')
   plt.ylabel('Value')
   plt.legend()
   plt.tight_layout()

   # Sauvegarder le graphique dans un fichier temporaire
   with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_image:
      plt.savefig(tmp_image.name, format='png')
      tmp_image_path = tmp_image.name

   # Générer le PDF
   pdf = FPDF()
   pdf.add_page()
   pdf.set_font("Arial", 'B', 12)
   pdf.cell(200, 10, 'Petroleum Market Trends in the USA', ln=True, align='C')

   # Insérer le graphique dans le PDF
   pdf.image(tmp_image_path, x=10, y=25, w=190)
   pdf.ln(105)

   # Ajouter le texte d’analyse
   pdf.set_font("Arial", '', 10)
   analysis = (
      "GDP reflects the level of economic activity and energy consumption. When the economy grows, energy demand rises, often pushing petroleum prices upward. In contrast, economic downturns reduce demand, which may drive prices down.\n\n"
      "Imports influence domestic supply. An increase in oil imports can expand supply and reduce prices. Conversely, a decline in imports may cause prices to rise if local production can't compensate.\n\n"
      "Exports reflect international demand. When exports rise, domestic supply tightens, potentially increasing local prices. Lower exports may ease domestic availability, stabilizing or lowering prices.\n\n"
      "Oil production is central to supply dynamics. Higher production tends to lower prices by increasing availability. Conversely, reduced output may lead to price hikes. Tracking production trends is key to understanding price movements.\n\n"
      "Inflation affects petroleum prices through rising costs and currency value changes. Higher inflation often increases energy production and transport costs, pushing oil prices upward. It also impacts interest rates and exchange rates, both of which influence oil pricing.\n\n"
      "Global events—including conflicts, economic crises, and policy decisions—can abruptly disrupt supply or demand. These shifts often lead to significant oil price fluctuations, making geopolitical and economic awareness essential for understanding petroleum market trends."
   )

   for paragraph in analysis.split('\n\n'):
      paragraph_clean = clean_text_for_pdf(paragraph)
      pdf.multi_cell(0, 8, paragraph_clean)
      pdf.ln(1)

   # Ajouter tableau résumé (moyennes annuelles)
   pdf.add_page()
   pdf.set_font("Arial", 'B', 10)
   col_widths = [20, 20, 20, 20, 25, 20, 25]
   headers = ['Year', 'Price', 'Import', 'Export', 'Production', 'Inflation', 'GDP']
   for i, header in enumerate(headers):
      pdf.cell(col_widths[i], 8, header, border=1)
   pdf.ln()

   pdf.set_font("Arial", '', 8)
   for row in df_annual.itertuples(index=False):
      for i, value in enumerate(row):
         pdf.cell(col_widths[i], 8, f"{value:.2f}" if isinstance(value, float) else str(value), border=1)
      pdf.ln()

   # Retourner le PDF
   pdf_output = BytesIO()
   pdf.output(pdf_output)
   pdf_output.seek(0)
   
   # Après l’envoi du fichier PDF
   os.remove(tmp_image_path)

   return send_file(pdf_output, as_attachment=True, download_name='petroleum_trends.pdf', mimetype='application/pdf')


def scrape_oilprice_news():
   url = "https://oilprice.com/Latest-Energy-News/World-News/"
   headers = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
   }

   response = requests.get(url, headers=headers)
   soup = BeautifulSoup(response.text, 'html.parser')

   news_items = []
   article_links = soup.find_all('a', href=True)

   for link in article_links:
      title_tag = link.find('h2', class_='categoryArticle__title')
      if title_tag:
         title = title_tag.get_text(strip=True)
         href = link['href']
         parent = link.parent
         excerpt_tag = parent.find('p', class_='categoryArticle__excerpt')
         excerpt = excerpt_tag.get_text(strip=True) if excerpt_tag else 'No excerpt available'

         news_items.append({
            'title': title,
            'href': href,
            'excerpt': excerpt
         })

   return news_items
@app.route('/events')
def eventpage():
   news_items = scrape_oilprice_news()
   return render_template('event.html', news_items=news_items)

@app.route('/')
def homepage():
   return render_template('index.html')


def scheduled_job():
   print("Running scheduled data update...")
   append_new_data_to_excel(r'C:\Users\USER\PycharmProjects\PlatformePFS\model\petrole_ai.xlsx')


scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_job, 'cron', day=1, hour=0, minute=0)  # 12:00 AM on the first day of every month
scheduler.start()

if __name__ == '__main__':
   app.run(debug=True, host='127.0.0.1', port=5001)
