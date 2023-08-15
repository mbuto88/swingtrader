import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l2
import numpy as np
import datetime
from stockprediction import StockPrediction
from keras.layers import Bidirectional, LSTM, Conv1D, Attention, Concatenate, Dropout
from keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np


def build_model_advanced(input_shape, output_shape, lstm_units, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_shape))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def AdvancedScaleDataBuildModelV2(symbol, filename):
    try:
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])

        if len(df) < 2 or df.iloc[1].isnull().all():
            print("The second row is empty or the dataframe doesn't have enough rows.")
            return StockPrediction(symbol, 1, 0, 0, 0, 0, 0)

        # Create separate scaler for input features
        input_scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit and transform the input scaler
        scaled_input = input_scaler.fit_transform(df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]])

        # Reshape data for LSTM [samples, timesteps, features]
        X, y, times = [], [], []
        for i in range(60, len(scaled_input)):
            X.append(scaled_input[i - 60:i])
            y.append(df[["Open", "High", "Low", "Close"]].iloc[i])
            times.append(df['Date'].iloc[i])
        X, y = np.array(X), np.array(y)

        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            time_test = [times[i] for i in test_index]

            # Create output scaler and fit on y_train only
            output_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_scaled = output_scaler.fit_transform(y_train)
            y_test_scaled = output_scaler.transform(y_test)

            # Build and train the model
            model = build_model_advanced((X_train.shape[1], X_train.shape[2]), y_train_scaled.shape[1], 50, 0.2, 0.01)
            model.fit(X_train, y_train_scaled, epochs=50, batch_size=32)

            # Rest of the code continues as before...

    except Exception as e:
        print(f"An error occurred: {e}")
        return StockPrediction(symbol, "1/1/1", 0, 0, 0, 0, 0)


def scaleDataBuildModelV2(symbol, filename):
    try:
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])

        if len(df) < 2 or df.iloc[1].isnull().all():
            print("The second row is empty or the dataframe doesn't have enough rows.")
            return StockPrediction(symbol, 1, 0, 0,0,0,0)

        # Create separate scalers for input and output features
        input_scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit and transform the input scaler
        scaled_input = input_scaler.fit_transform(df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]])

        # Reshape data for LSTM [samples, timesteps, features]
        X, y, times = [], [], []
        for i in range(60, len(scaled_input)):
            X.append(scaled_input[i - 60:i])
            y.append(df[["Open", "High", "Low", "Close"]].iloc[i])
            times.append(df['Date'].iloc[i])
        X, y = np.array(X), np.array(y)

        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            time_test = [times[i] for i in test_index]

            # Create output scaler and fit on y_train only
            output_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_scaled = output_scaler.fit_transform(y_train)
            y_test_scaled = output_scaler.transform(y_test)

            # Build and train the model
            model = build_model((X_train.shape[1], X_train.shape[2]), y_train_scaled.shape[1], 50, 0.2, 0.01)
            model.fit(X_train, y_train_scaled, epochs=50, batch_size=32)

            # Predict the next day's stock prices
            last_60_days = scaled_input[-60:]
            next_day_prediction = model.predict(last_60_days[np.newaxis, :, :])
            next_day_prediction_transformed = output_scaler.inverse_transform(next_day_prediction)

            # Get the time for the next day's prediction
            next_day_time2 = df['Date'].iloc[-1] + pd.DateOffset(days=1)
            next_day_time_in_human_format2 = next_day_time2.strftime('%m/%d/%Y')

            # Display the next day's prediction with its corresponding time
            print(f"Date: {next_day_time_in_human_format2}, Predicted Open: {next_day_prediction_transformed[0][0]}, High: {next_day_prediction_transformed[0][1]}, Low: {next_day_prediction_transformed[0][2]}, Close: {next_day_prediction_transformed[0][3]}")
            day_delta = next_day_prediction_transformed[0][3] - next_day_prediction_transformed[0][0]

        return StockPrediction(symbol, next_day_time_in_human_format2, day_delta, next_day_prediction_transformed[0][1],
                               next_day_prediction_transformed[0][0], next_day_prediction_transformed[0][3],
                               next_day_prediction_transformed[0][0])
    except Exception as e:
        print(f"An error occurred: {e}")
        return StockPrediction(symbol, "1/1/1", 0, 0, 0, 0, 0)


def build_model(input_shape, output_shape, neurons, dropout, decay):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(output_shape, kernel_regularizer=l2(decay)))
    model.compile(loss='mse', optimizer='adam')
    return model

def unix_timestamp_to_mmddyyyy(unix_timestamp):
    # Convert Unix timestamp to datetime object
    dt_object = datetime.datetime.fromtimestamp(unix_timestamp)

    # Format the datetime object as mm/dd/yyyy
    formatted_date = dt_object.strftime("%m/%d/%Y")

    return formatted_date

