import os
import csv
from datetime import datetime

#StockCSVObj("test","N:\stockdata\\test.csv")

def clean_csv_list(stockCsvObjects):
    clean_list = []
    for stock in stockCsvObjects:
        with open(stock.filename, 'r') as file:
            reader = csv.reader(file)
            lines = list(reader)
            if len(lines) <= 1:
                print(f'{stock.symbol} File {stock.filename} has been deleted.')
            else:
                print(f'{stock.symbol} File {stock.filename} contains more than one line.')
                clean_list.append(stock)

    return clean_list

def write_stock_predictions_to_csv(predictions):
    # Get today's date
    today = datetime.today().strftime('%m-%d-%Y')

    # Define the CSV filename
    filename = f'N:\stockdata\FinalPredictions\sorted-picks-{today}.csv'

    # Define the column headers for the CSV
    headers = ['symbol', 'date', 'day_delta', 'high', 'open_price', 'close_price', 'adjusted_close', 'percentage_delta', 'dollar_delta']

    # Write the objects to the CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        writer.writeheader()
        for prediction in predictions:
            writer.writerow({
                'symbol': prediction.symbol,
                'date': prediction.date,
                'day_delta': prediction.day_delta,
                'high': prediction.high,
                'open_price': prediction.open,
                'close_price': prediction.close,
                'adjusted_close': prediction.adjustedClose,
                'percentage_delta': prediction.percentage_delta,
                'dollar_delta': prediction.dollar_delta
            })

    print(f"File '{filename}' has been written successfully.")