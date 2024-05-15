import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
