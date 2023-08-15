import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from stockcsvobj import StockCSVObj
from convertcsvtounixtimestamp import ConvertToUnixTimestamp, addHeaders
import os

def gethistoricaldata(stocks):
    test = StockCSVObj("test","N:\stockdata\\test.csv")
    listofhistoricaldatacsvs = [test]

    # Define the data range
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')

    # Download the data
    for stock in stocks:
        csvname = f"N:\stockdata\\{stock}_5_years.csv"
        if not os.path.isfile(csvname):
            data = yf.download(stock, start=start_date, end=end_date)
            print(f"now getting historical data for: {stock}")

            data.to_csv(csvname)
            addHeaders(csvname)
            unix_csv_name = ConvertToUnixTimestamp(csvname, stock)
            listofhistoricaldatacsvs.append(StockCSVObj(stock, unix_csv_name))
        else:
            unix_csv_name = ConvertToUnixTimestamp(csvname, stock)

            listofhistoricaldatacsvs.append(StockCSVObj(stock, unix_csv_name))

            print(f"Stock data for {stock} exists")

    return listofhistoricaldatacsvs