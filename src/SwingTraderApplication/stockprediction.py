class StockPrediction:
    def __init__(self, symbol, date, day_delta, high, open_price, close_price, adjusted_close):
        self.symbol = symbol
        self.date = date
        self.day_delta = day_delta #change in price through the day
        self.high = high
        self.open = open_price
        self.close = close_price
        self.adjustedClose = adjusted_close

# Creating an instance of the StockPrediction class
# prediction1 = StockPrediction("AAPL", "2023-07-30", 148.50, 150.25, 149.75)

    @property
    def percentage_delta(self):
        if(self.open != 0):
            return ((self.high - self.open) / self.open) * 100
        else:
            return 0;

    @property
    def dollar_delta(self):
        return (self.high - self.open)

    def toString(self):
        string = (f"Date:{self.date} symbol:{self.symbol} day_delta:{self.day_delta} high:{self.high} close:{self.close}")
        print(string)
        return string;