import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Préparation des données
def prepare_data(file_path='model/petrole_ai.xlsx', seq_length=12):
    df = pd.read_excel(file_path, header=1)
    df.columns = ['Date','Prix', 'Import (Thousand Barrels )', 'Export',
                  'Production(Thousand Barrels per Day)', 'Inflation (%)',
                  'GDP(Billions of USD)', 'Event']

    # OneHot Encoding de la colonne 'Event'
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    event_encoded = encoder.fit_transform(df[['Event']])
    event_df = pd.DataFrame(event_encoded, index=df.index, columns=encoder.get_feature_names_out(['Event']))
    df = pd.concat([df.drop('Event', axis=1), event_df], axis=1)

    # Normalisation (hors Date)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.drop('Date', axis=1))

    # Création des séquences
    def create_sequences(data, target_index, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i])
            y.append(data[i, target_index])
        return np.array(X), np.array(y)

    target_index = list(df.drop('Date', axis=1).columns).index('Prix')
    X, y = create_sequences(scaled_data, target_index, seq_length)
    return X, y, scaler, target_index, scaled_data

# Entraînement du modèle
def train_model(X, y, target_index, scaled_data, seq_length=12):
    split_idx = int(0.85 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Définition du modèle
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )

    # Prédictions
    y_pred = model.predict(X_test)

    # Inverser la normalisation (pour le prix)
    y_test_actual = scaler.inverse_transform(
        np.concatenate([
            np.zeros((len(y_test), target_index)),
            y_test.reshape(-1, 1),
            np.zeros((len(y_test), scaled_data.shape[1] - target_index - 1))
        ], axis=1)
    )[:, target_index]

    y_pred_actual = scaler.inverse_transform(
        np.concatenate([
            np.zeros((len(y_pred), target_index)),
            y_pred,
            np.zeros((len(y_pred), scaled_data.shape[1] - target_index - 1))
        ], axis=1)
    )[:, target_index]

    # === Calcul des métriques ===
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mean_y = np.mean(y_test_actual)

    if mean_y > 1e-3:
        precision = 100 - (mae / mean_y) * 100
    else:
        precision = 0
    precision = round(precision, 2)

    if precision >= 90:
        commentaire = "Excellente précision. Le modèle LSTM capture très bien les dynamiques du marché."
    elif precision >= 80:
        commentaire = "Bonne précision. Quelques fluctuations peuvent ne pas être bien anticipées."
    elif precision >= 70:
        commentaire = "Précision moyenne. Le modèle pourrait être amélioré avec plus de données ou un réglage fin."
    else:
        commentaire = "Précision faible. Le modèle a des difficultés à bien prédire les tendances du marché."

    print("\n=== Évaluation du modèle ===")
    print(f"Précision : {precision}%")
    print(f"MAE (Erreur absolue moyenne) : {mae:.2f}")
    print(f"RMSE (Erreur quadratique moyenne) : {rmse:.2f}")
    print(f"Prochaine valeur prédite : {y_pred_actual[-1]:.2f}")
    print(f"Commentaire global : {commentaire}")

    return model, scaler, target_index, scaled_data, X_test, y_test, y_test_actual, y_pred_actual

# Sauvegarde
def save_all(model, scaler, target_index, scaled_data, X_test, y_test, model_dir='model'):
    os.makedirs(model_dir, exist_ok=True)
    model.save(f'{model_dir}/lstm_model.h5')
    joblib.dump(scaler, f'{model_dir}/scaler_lstm.pkl')
    joblib.dump(target_index, f'{model_dir}/target_index.pkl')
    joblib.dump(scaled_data, f'{model_dir}/scaled_data.pkl')
    np.save(f'{model_dir}/x_test_lstm.npy', X_test)
    pd.DataFrame(y_test).to_csv(f'{model_dir}/y_test_lstm.csv', index=False)
    print("Modèle, scaler, données test, et fichiers sauvegardés avec succès.")

# === MAIN ===
if __name__ == "__main__":
    seq_length = 12
    X, y, scaler, target_index, scaled_data = prepare_data(seq_length=seq_length)
    model, scaler, target_index, scaled_data, X_test, y_test, y_test_actual, y_pred_actual = train_model(
        X, y, target_index, scaled_data, seq_length=seq_length)
    save_all(model, scaler, target_index, scaled_data, X_test, y_test)
