import pandas as pd

def ConvertToUnixTimestamp(csvname, stock):
    # Load the CSV file
    df = pd.read_csv(csvname, skiprows=1)
    print(f"Converting {csvname} to unix timestamp")
    # Iterate over the rows of the DataFrame
    for i, row in df.iterrows():

        # Convert the date to datetime format
        original_date = row[0]
        converted_date = pd.to_datetime(original_date, format='%Y/%m/%d')

        # Convert the date to Unix epoch time
        unix_time = (converted_date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        # Update the date in the DataFrame
        df.at[i, 'Date'] = unix_time

    unix_csvname = f"N:\stockdata\\{stock}_5_years-unix-timestamp.csv"
    # Save the modified DataFrame back to a CSV file
    df.to_csv(unix_csvname, index=False)

    return unix_csvname

def addHeaders(unix_csvname):

    # Define the column names
    column_names = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

    # Read the CSV file without headers
    df = pd.read_csv(unix_csvname, header=None)

    # Set the column names
    df.columns = column_names

    # Write the DataFrame back to the CSV file
    df.to_csv(unix_csvname, index=False)