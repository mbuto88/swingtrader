import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

def modelV2(csv_name):
    # Load the dataset
    df = pd.read_csv(csv_name)

    # We'll use the 'Close' prices for our predictions
    data = df['Close'].values
    data = data.astype('float32')

    # Normalize the data to help the LSTM model learn more effectively
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare the sequences for our LSTM model
    timesteps = 60

    X = []
    y = []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i, 0])
        y.append(data[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshape the data to 3D for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split the data into train and test sets
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile and fit the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # Predict the prices
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot the results
    plt.figure(figsize=(8, 4))
    plt.plot(y_test, color='blue', label='Actual Stock Price')
    plt.plot(predictions , color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
