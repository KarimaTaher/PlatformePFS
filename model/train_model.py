import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

def prepare_data(file_path='model/petrole_ai.xlsx', seq_length=24):
    df = pd.read_excel(file_path, header=1)
    df.columns = ['Date','Prix', 'Import (Thousand Barrels )', 'Export',
                'Production(Thousand Barrels per Day)', 'Inflation (%)',
                'GDP(Billions of USD)', 'Event']

    # Encodage OneHot de la colonne "Event"
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    event_encoded = encoder.fit_transform(df[['Event']])
    event_df = pd.DataFrame(event_encoded, index=df.index, columns=encoder.get_feature_names_out(['Event']))
    df = pd.concat([df.drop('Event', axis=1), event_df], axis=1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.drop('Date', axis=1))

    def create_sequences(data, target_index, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i, target_index])
        return np.array(X), np.array(y)

    target_index = list(df.drop('Date', axis=1).columns).index('Prix')
    X, y = create_sequences(scaled_data, target_index, seq_length)
    return X, y, scaler, target_index, scaled_data

# Préparation des données
seq_length = 24
X, y, scaler, target_index, scaled_data = prepare_data(seq_length=seq_length)

# Séparation entraînement/test : 70% / 30%
split_idx = int(0.7 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Définition du modèle
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu'))
model.add(Dense(1))  # Couche de sortie

model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement
model.fit(X_train, y_train, epochs=83, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Sauvegarde
model.save('model/lstm_model.h5')
joblib.dump(scaler, 'model/scaler.save')
joblib.dump(target_index, 'model/target_index.save')
joblib.dump(scaled_data, 'model/scaled_data.save')

print("Modèle et fichiers sauvegardés avec succès.")
