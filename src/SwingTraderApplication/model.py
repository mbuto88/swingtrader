import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l2
import numpy as np
import datetime
from stockprediction import StockPrediction

def scaleDataBuildModelV2(symbol, filename):
    try:
        df = pd.read_csv(filename)
        if len(df) < 2 or df.iloc[1].isnull().all():
            print("The second row is empty or the dataframe doesn't have enough rows.")
            return StockPrediction(symbol, 1, 0, 0,0,0,0)

        # Convert 'Date' into separate features
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day

        # Create separate scalers for input and output features
        input_scaler = MinMaxScaler(feature_range=(0, 1))
        output_scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit the scalers
        scaled_input_features = ["Year", "Month", "Day", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        scaled_output_features = ["Open", "High", "Low", "Close"]
        scaled_input = input_scaler.fit_transform(df[scaled_input_features])
        scaled_output = output_scaler.fit_transform(df[scaled_output_features])

        # Reshape data for LSTM [samples, timesteps, features]
        X, y, times = [], [], []
        for i in range(60, len(scaled_input)):
            X.append(scaled_input[i - 60:i])
            y.append(scaled_output[i])  # Predicting 'open', 'high', 'low', 'close'
            times.append(df['Date'].iloc[i])
        X, y = np.array(X), np.array(y)

        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            time_test = [times[i] for i in test_index]

            # Build and train the model
            model = build_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1], 50, 0.2, 0.01)
            model.fit(X_train, y_train, epochs=50, batch_size=32)

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

