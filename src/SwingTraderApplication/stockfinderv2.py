import requests
from bs4 import BeautifulSoup
import csv
from stockcsvobj import StockCSVObj
import string

import yfinance as yf

def fetchStocksByMarketv2(market):
    symbols = scrape_symbols(market)
    valid_symbols = []

    with open(f"N:\\stockdata\\{market}.csv", "w") as f:
        writer = csv.writer(f)
        for symbol in symbols:
            # Fetch P/E ratio, Beta, and EPS
            pe_ratio, beta, eps = get_factors(symbol)

            # Only add the symbol to the CSV if all three values are found
            if pe_ratio is not None and beta is not None and eps is not None:
                writer.writerow([symbol])
                valid_symbols.append(symbol)

    print(f"Completed fetching all valid stocks for market: {market}")
    return valid_symbols

def get_factors(stock_symbol):
    print(f"Getting factors for {stock_symbol}")
    try:
        stock = yf.Ticker(stock_symbol)
        pe_ratio = stock.info['trailingPE']
        beta = stock.info['beta']
        eps = stock.info['trailingEps']
        #print("\033[92mGreen text!\033[0m")

        #print(f"factors found are: p/e ratio: {pe_ratio} beta: {beta} eps: {eps}")
        print(f"\033[92mfactors found are: p/e ratio: {pe_ratio} beta: {beta} eps: {eps}\033[0m")
        return pe_ratio, beta, eps
    except:
        return None, None, None

def select_best_stocks(stock_symbols, top_n=150, weights=(0.5, 0.3, 0.2)):
    stock_factors = {}
    print(f"Starting to select top:{top_n} best stocks from list given")

    # Fetch the P/E ratios, Beta, and EPS (earnings per share) for the given stock symbols
    for symbol in stock_symbols:
        pe_ratio, beta, eps = get_factors(symbol)
        print("Types:", type(pe_ratio), type(beta), type(eps))
        print("Values:", pe_ratio, beta, eps)
        print(f"checking validity of data for {symbol}")
        # Check if any of the variables are not numbers before calculating the score
        if isinstance(pe_ratio, (int, float)) and isinstance(beta, (int, float)) and isinstance(eps, (int, float)):
            if pe_ratio is not None and beta is not None and eps is not None:
                print(f"getting score for {symbol}")
                # Weighted score calculation
                score = weights[0] * pe_ratio + weights[1] * beta - weights[2] * eps
                stock_factors[symbol] = score
        else:
            print(f"Unexpected types for symbol {symbol}. Skipping.")

    # Sort the stocks by the weighted score and select the top N
    sorted_stocks = sorted(stock_factors.items(), key=lambda x: x[1])
    best_stocks = sorted_stocks[:top_n]

    return [stock[0] for stock in best_stocks]

def scrape_symbols(exchange):

    alphabet = list(string.ascii_lowercase)
    symbols = []
    i = 0

    for letter in alphabet:
        #https://eoddata.com/stocklist/NASDAQ/B.htm
        print(f"Getting stock symbols for letter {letter} from {exchange}")
        url = f"http://eoddata.com/stocklist/{exchange}/{letter}.htm"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")

        # find the table with the stock symbols
        table = soup.find('table', {'class': 'quotes'})

        if table is not None:
            # find all table rows
            rows = table.findAll('tr')

            for row in rows:
                # find each cell in the row
                cells = row.findAll('td')

                # append the first cell (the symbol) to the list if it exists
                if len(cells) > 0:
                    symbols.append(cells[0].text.rstrip())
                #i+=1
                #if i == 10:
                    #return symbols

    return symbols

def fetchStocksByMarketAndReduceSizeOfList(market, listSize):
    allSymbols = fetchStocksByMarketv2(market)
    return select_best_stocks(allSymbols, listSize)

def fetchStocksByMarket(market):
    symbols = scrape_symbols(market)

    with open(f"N:\stockdata\\{market}.csv", "w") as f:
        writer = csv.writer(f)
        for symbol in symbols:
            writer.writerow([symbol])
    print(f"Completed fetching all stocks for market:{market}")
    return symbols

def findstocks():
    nasdaq_symbols = scrape_symbols("NASDAQ")
    nyse_symbols = scrape_symbols("NYSE")

    with open("N:\stockdata\\nasdaq_symbols.csv", "w") as f:
        writer = csv.writer(f)
        for symbol in nasdaq_symbols:
            writer.writerow([symbol])

    with open("N:\stockdata\\nyse_symbols.csv", "w") as f:
        writer = csv.writer(f)
        for symbol in nyse_symbols:
            writer.writerow([symbol])

    return nasdaq_symbols, nyse_symbols