# model/lstm_model.py

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def predict_next_month_price(file_path='model/petrole_ai.xlsx'):
    df = pd.read_excel(file_path, header=1)
    df.columns = ['Date','Prix', 'Import (Thousand Barrels )', 'Export',
                'Production(Thousand Barrels per Day)', 'Inflation (%)',
                'GDP(Billions of USD)', 'Event']

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    event_encoded = encoder.fit_transform(df[['Event']])
    event_df = pd.DataFrame(event_encoded, index=df.index, columns=encoder.get_feature_names_out(['Event']))
    df = pd.concat([df.drop('Event', axis=1), event_df], axis=1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.drop('Date', axis=1))

    def create_sequences(data, target_index, seq_length=12):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i, target_index])
        return np.array(X), np.array(y)

    target_index = list(df.drop('Date', axis=1).columns).index('Prix')
    X, y = create_sequences(scaled_data, target_index)

    split_idx = int(0.85 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=0)

    y_pred = model.predict(X_test)

    # Dernière prédiction
    last_input = scaled_data[-12:]
    next_input = np.expand_dims(last_input, axis=0)
    next_pred_scaled = model.predict(next_input)

    next_pred = scaler.inverse_transform(np.concatenate([
        np.zeros((1, target_index)),
        next_pred_scaled,
        np.zeros((1, scaled_data.shape[1] - target_index - 1))
    ], axis=1))[:, target_index][0]

    return round(next_pred, 2)
