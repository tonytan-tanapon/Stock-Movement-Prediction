import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import matplotlib.pyplot as plt

def load_data():
    test_data = pd.read_csv('data/test_data.csv', parse_dates=['Date'])
    X_test = test_data[['Open', 'High', 'Low', 'Volume']].values
    y_test = test_data['Close'].values
    return X_test, y_test

def evaluate_model():
    X_test, y_test = load_data()
    
    scaler = load('results/scaler.joblib')
    X_test_scaled = scaler.transform(X_test)
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1))
    
    model = tf.keras.models.load_model('results/model.h5')
    predictions = model.predict(X_test_scaled)
    
    y_test_inv = scaler.inverse_transform(y_test_scaled)
    predictions_inv = scaler.inverse_transform(predictions)
    
    plt.plot(y_test_inv, label='Actual Prices')
    plt.plot(predictions_inv, label='Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('results/predictions.png')
    
    mse = np.mean((y_test_inv - predictions_inv)**2)
    with open('results/evaluation_results.txt', 'w') as f:
        f.write(f'Mean Squared Error: {mse}\n')

if __name__ == '__main__':
    evaluate_model()
