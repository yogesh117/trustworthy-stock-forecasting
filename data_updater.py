
# Team 2
# Olin Dsouza (odsouza1@stevens.edu), Yogesh Patil (ypatel33@stevens.edu), Kevin Gwinn (kgwinn@stevens.edu)
# Date: 8/16/26
# Description: Engineering Python Final Project, Trustworthy Stock Forecaster
# File name: data_updater.py

"""
data_updater.py module is responsible for calling an API and getting new data from the stock market. By default it calls daily time series
which returns a date, volume, high, low, open and close for the history of the stock.

add_update_symbol takes a stock symbol, and creates a csv of up to date data if it does not exist, and updates the csv file for that symbol 
if it does already exist

cleanup_symbols looks through the data folder and finds files that are too small to hold valuable data and removes them

remove_symbol takes a stock symbol as an argument and removes its csv from the data folder so it will not be inculded in any processing.
it checks the request to remove to make sure that sample files are not removed.

"""

import urllib.request
import urllib.error
import os
import pandas as pd
from pathlib import Path

#the API key is read from the environment so no secret is committed to the repository.
#set it before running:  export ALPHAVANTAGE_API_KEY=yourkey   (Mac)  /  $env:ALPHAVANTAGE_API_KEY="yourkey"  (Windows)
#this module is an OPTIONAL utility - the main program runs entirely from the committed sample CSVs.
API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "demo")

def cleanup_symbols():  
    """
    Iterates through the data folder and checks the size of the csv files. if a file has less than 10 rows it will be removed.
    Under 10 rows means that either the API call has failed (unsupported symbol, or no internet connection, etc.), or that the stock
    symbol simply did not return enough data to be calculated.
    """
    target_dir = Path(__file__).parent / "data"        #set path to the data folder
    for f in target_dir.iterdir():                     #iterate through data folder
        if f.is_file():
            df = pd.read_csv(f)                        #read file into panda dataframe
            row_count = len(df)                        #get number of df rows
            
            if row_count < 10:
                 if os.path.exists(f):                 #remove the file if it exists and is under 10 rows
                    os.remove(f)
                 

def add_update_symbol(symbol='AAPL'):
    """
    takes a stock symbol and calls an API to update the stocks daily information with the most recent information

    Parameters
    symbol : str (default 'AAPL')
        stock symbol available for trading on major US stock exchanges

    Output:
        csv file with latest 1000 days of data formated for data_loader is written to the data folder
    """

    function_call = 'TIME_SERIES_DAILY'           #This returns daily information, can be changed for more granular data

    #url for API call using alphavantage's API
    url = f'https://www.alphavantage.co/query?function={function_call}&symbol={symbol}&datatype=csv&outputsize=full&apikey={API_KEY}'

    temp = Path(__file__).parent / "data" / f"temp.csv"              #the API returns csv formatted data which will be stored in a temp file first
    output = Path(__file__).parent / "data" / f"{symbol}.csv"        #if the data is valid, it will be used to update or create the output file with symbol name

    try:                                                    #try/except to make sure that a file is created
        if os.path.exists(temp):                            #remove temp file if it does exist
            os.remove(temp)

        with urllib.request.urlopen(url) as response:       #call url and read/decode data
            data = response.read().decode("utf-8")

        with open(temp, "w", encoding="utf-8") as file:     #write the data to a csv file cleaned up with and newline characters stripped out
            for line in [data]:
                cleaned_line = line.rstrip("\n")
                if len(cleaned_line) != 0:                  #if the line is not empty it is printed to the file
                    print(cleaned_line, file=file)
            
        df = pd.read_csv(temp)                              #put the temp file into a panda dataframe
        df = df.head(1000)                                  #take the most recent 1000 rows
        df = df.rename(columns={"timestamp": "Date",        #rename columns to match desired labels
                                "high":"High",
                                "low":"Low",
                                "open":"Open",
                                "close":"Close",
                                "volume":"Volume"})   

        if len(df) > 10:                                   #check that the file has at least 10 rows of data (so no failed API calls are added)
            df.to_csv(output, index=False)                 #if there is enough data create or update the the stock csv data

        if os.path.exists(temp):                           #if temp file still exists remove it from the data folder
            os.remove(temp)
 
    except (urllib.error.URLError, OSError, pd.errors.ParserError) as error:    #catch network/file/parse failures and notify user
        print(f"something went wrong, no file created: {error}")

def remove_symbol(symbol):
    """
    takes a stock symbol string searches the data folder for the symbol's csv folder and removes it if it exists. This allows for efficiency if portfolio
    building by excluding unwanted symbols.

    Parameters
    symbol : str 
        stock symbol that will be removed from data folder

    Output:
        if the csv file exists it is removed from the data folder
    """
    target = Path(__file__).parent / "data" / f"{symbol}.csv"        #set path for csv that will be looked for
    if symbol not in ["aapl_sample", "jpm_sample","msft_sample"]:    #check that symbol is a sample file do NOT allow sample files to be removed
        if os.path.exists(target):                                   #if the csv file exists and is not a sample, it is removed
            os.remove(target)

if __name__ == "__main__":    #allow module to run on its own to test
    add_update_symbol('IBM')  #IBM is a known good symbol that should be able to be added to the data folder succesfully
    #cleanup_symbols()        #uncomment to test cleanup
    #remove_symbol('IBM')     #uncomment to remove the IBM.csv file 