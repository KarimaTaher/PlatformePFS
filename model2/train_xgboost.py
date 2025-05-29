# /model2/train_xgboost.py

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import os

# Charger les données
file_path = 'model2/petrole_ai.xlsx'  # Mets ton chemin correct si tu l'exécutes ailleurs
df = pd.read_excel(file_path, header=1)

df.columns = ['Date','Prix', 'Import (Thousand Barrels )', 'Export',
            'Production(Thousand Barrels per Day)', 'Inflation (%)',
            'GDP(Billions of USD)', 'Event']

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Lags et moyennes glissantes
df['Prix_t-1'] = df['Prix'].shift(1)
df['Prix_t-2'] = df['Prix'].shift(2)
df['Prix_t-3'] = df['Prix'].shift(3)
df['Prix_roll_mean_3'] = df['Prix'].rolling(window=3).mean()
df['Prix_roll_mean_6'] = df['Prix'].rolling(window=6).mean()
df['Prix_roll_mean_12'] = df['Prix'].rolling(window=12).mean()

df.dropna(inplace=True)

# Encodage de Event
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
event_encoded = encoder.fit_transform(df[['Event']])
event_encoded_df = pd.DataFrame(event_encoded, columns=encoder.get_feature_names_out(['Event']))

df_encoded = pd.concat([df.reset_index(drop=True), event_encoded_df], axis=1)

X = df_encoded.drop(['Date', 'Prix', 'Event'], axis=1)
y = df_encoded['Prix']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modèle
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=5,
    random_state=42,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# Sauvegarde du modèle et de l'encoder
os.makedirs("model2", exist_ok=True)
joblib.dump(model, "model2/xgb_model.pkl")
joblib.dump(encoder, "model2/xgb_encoder.pkl")
X_test.to_csv("model2/x_test.csv", index=False)
y_test.to_csv("model2/y_test.csv", index=False)

print("✅ Modèle et données sauvegardés.")
