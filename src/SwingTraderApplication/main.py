import pandas as pd
from model import scaleDataBuildModel, scaleDataBuildModelV2
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l2
from stockfinderv2 import findstocks, fetchStocksByMarket, fetchStocksByMarketAndReduceSizeOfList, fetchStocksByMarketv2
import numpy as np
import datetime
from historicaldata import gethistoricaldata
from modelv2 import modelV2
from stockcsvobj import StockCSVObj
from stockprediction import StockPrediction
import tensorflow as tf
from data_management import clean_csv_list, write_stock_predictions_to_csv
import csv
import keyboard
import time
from datetime import datetime
import threading

class Logger:
    def __init__(self, file_name):
        self.log_file = open(file_name, 'w')
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.log_file.flush()

def write_to_csv(predictions):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = f'N:\\stockdata\\snapshot_results_{timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(predictions.toString)
    print(f'Snapshot saved to {filename}')

def listen_for_keypress(array):
    print("Press 'p' to take a snapshot of the array and save to CSV")
    while True:
        if keyboard.is_pressed('p'):
            write_to_csv(array)
            time.sleep(1) # Prevents multiple snapshots if 'p' is held down

def sortByDayDelta(stock_predictions):
    stock_predictions.sort(key=lambda x: x.day_delta, reverse=True)
    return stock_predictions

def sortByPercentageDelta(stock_predictions):
    stock_predictions.sort(key=lambda x: x.percentage_delta, reverse=True)
    return stock_predictions

def sortByDollarDelta(stock_predictions):
    stock_predictions.sort(key=lambda x: x.dollar_delta, reverse=True)
    return stock_predictions

def getDateTime():
    current_date_time = datetime.now()
    return current_date_time.strftime("%Y-%m-%d %H:%M:%S")

def main():
    sys.stdout = Logger('output_log.txt')

    print("This will be printed to the console and written to the file")
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # Get list of stock names
    # nyse, nasdaq = findstocks()
    #symbols = nyse + nasdaq

    #symbols = fetchStocksByMarket("nasdaq")
    symbols = fetchStocksByMarketAndReduceSizeOfList("nasdaq", 40)
    # get price data for stocks (5 years historical data)
    historicaldata = gethistoricaldata(symbols)
    # print("\033[92mGreen text!\033[0m")
    print(f"\033[92m{getDateTime()}Completed retrieving historical price data \n\033[0m")

    clean_historical_data = clean_csv_list(historicaldata)
    print(f"\033[92m{getDateTime()}Completed cleaning list of historical price data csvs and deleted empty csv files \n\033[0m")

    for stock in clean_historical_data:
        if stock.symbol == "test":
            continue
        print(stock.symbol, stock.filename)

    stock_price_predictions = []

    # listens for keypress 'p' by running on a separate thread,
    # really slows down how fast it runs for prediction calculations
    # threading.Thread(target=listen_for_keypress, args=(stock_price_predictions,)).start()

    for stock in historicaldata:
        if stock.symbol == "test":
            continue
        print(f"{getDateTime()}now predicting price for {stock.symbol}, from {stock.filename}\n")

        # train model on price and store price in array
        stock_price_predictions.append(scaleDataBuildModelV2(stock.symbol, stock.filename))
        print(f"\033[92m{getDateTime()}Completed prediction for {stock.symbol}")
        #stock_price_predictions.append(modelV2(stock.symbol, stock.filename))

    # Select most profitable stocks for the day, TODO: decide if choosing overall gain by price or percentage
    sortedPredictionsFromHighToLowByDayDelta = sortByDayDelta(stock_price_predictions)
    sortedPredictionsFromHighToLowByPercetageDelta = sortByPercentageDelta(stock_price_predictions)
    sortedPredictionsFromHighToLowByDollarDelta = sortByDollarDelta(stock_price_predictions)

    # Print best picks, by total increase
    print(f"\033[92m{getDateTime()}Predictions sorted by total dollar increase from open to close\033[0m")
    for prediction in sortedPredictionsFromHighToLowByDayDelta:
        print(prediction.symbol, prediction.day_delta)

    # Print best picks, by total increase
    print(f"\033[92m{getDateTime()}Predictions sorted by total dollar increase from open to high\033[0m")
    for prediction in sortedPredictionsFromHighToLowByDollarDelta:
        print(prediction.symbol, prediction.dollar_delta)

    # Print best picks, by total increase
    print(f"\033[92m{getDateTime()}Predictions sorted by total percent increase from open to close\033[0m")
    for prediction in sortedPredictionsFromHighToLowByPercetageDelta:
        print(prediction.symbol, prediction.percentage_delta)

    write_stock_predictions_to_csv(sortedPredictionsFromHighToLowByPercetageDelta)

    # TODO: buy picks from brokerage API with trailing stop


if __name__ == "__main__":
    main()