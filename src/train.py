import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import build_model
from joblib import dump

def load_data():
    train_data = pd.read_csv('data/train_data.csv', parse_dates=['Date'])
    X_train = train_data[['Open', 'High', 'Low', 'Volume']].values
    y_train = train_data['Close'].values
    return X_train, y_train

def train_model():
    X_train, y_train = load_data()
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    
    model = build_model(input_shape=(X_train_scaled.shape[1], 1))
    model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32)
    
    model.save('results/model.h5')
    dump(scaler, 'results/scaler.joblib')

if __name__ == '__main__':
    train_model()
