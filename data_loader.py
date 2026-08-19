# Team 2
# Olin Dsouza (odsouza1@stevens.edu), Yogesh Patil (yp36032@stevens.edu), Kevin Gwinn (kgwinn@stevens.edu)
# Date: 8/16/16
# Description: Engineering Python Final Project, Trustworthy Stock Forecaster
# File name: data_loader.py

"""
data_loader.py module is responsible for taking in CSV files with stock data, converting them to panda dataframes, 
then checking the data for completeness and cleaning up any missing data.

It then calculates the log of daily returns based on the close price each day, and calculates the volitility of a
period via a rolling 20 day period. it is then able to label the regime of that window as either calm or volatile 
"""

#import required libraries 
from pathlib import Path
import numpy as np
import pandas as pd



#a set will be used for the regime of the stocks. for this project only calm and volatile are used,
#but future improvements could expand regime categories with things like correlation, breadth, risk, speed of price changes, etc.
REGIME_LABELS = {"calm", "volatile"}


def load_daily_data(csv_path):
    """
    Loads data from a csv file given a path to the file, and checks for required columns of data

    Parameters
    csv_path : Path
        Path to a CSV file with columns of Date, Open, High, Low, Close, Volume columns.

    Returns
    DataFrame
        data sorted and indexed by date, with rows containing missing/invalid prices dropped.

    """
    csv_path = Path(csv_path)    #get the path to the csv file that will be read in and converted to dataframe
    if not csv_path.exists():    #check that there is a file at the given path
        raise FileNotFoundError(f"Price data file not found: {csv_path}")    #throw exception if there is no file at that path

    df = pd.read_csv(csv_path)        #convert csv to panda dataframe

    required_columns = ("Date", "Open", "High", "Low", "Close", "Volume")    #these are the minimum columns

    for c in required_columns:        #check that the dataframe has all required columns
        if c not in df.columns:
            raise ValueError(f"CSV is missing required columns: {c}")    #error if the column is missing

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")    #convert date column to date type, coerce error will keep from crashing
    for col in ("Open", "High", "Low", "Close", "Volume"):      #convert the rest of columns to numeric values
        df[col] = pd.to_numeric(df[col], errors="coerce")       #coerce errors will keep from crashing and substitute NaN for removal

    df = df.dropna(subset=["Date", "Close"])                    #drop any rows with NaN in date or close
    df = df[df["Close"] > 0]                                    #drop any rows with close less than 0 (negative close sometimes appear when exchange closes early)
    df = df.sort_values("Date").drop_duplicates(subset="Date")  #sort by date, remove duplicate dates
    df = df.set_index("Date")                                   #index set by date

    if len(df) == 0:
        raise ValueError(f"No usable rows remain in {csv_path}")   #if there were no viable rows throw an error
    return df


def compute_log_returns(df, close_price = "Close"):
    """
    Compute daily log-returns r_t = ln(P_t) - ln(P_<t-1>) <- formula and numpy implementation found on mlforanalytics.com

    Parameters
    df : pandas.DataFrame
        Price data in dataframe form
    close_price : str
        Column to compute returns on - defaults to close, could be changed if computing returns multiple times per day.

    Returns
    pd.Series
        Log-returns  with any NaN entries removed - the first entry always is dropped because it needs the entry before it.
    """

    if len(df) < 2:
        raise ValueError("Need at least 2 price data points to calculate logs")

    prices = df[close_price].astype(float)                   #dataframe of closing prices 
    log_returns = np.log(prices / prices.shift(1))           #calculate daily logs (formula taken from mlforanalytics.com )
    log_returns = pd.Series(log_returns, name="log_return")  #make log returns a pd series and name it
    log_returns = log_returns.dropna()                       #drop the NaN rows - the first row 
    
    return log_returns


def rolling_volatility(returns, window = 20):
    """
    Rolling standard deviation of log-returns over the period of a sliding window, default of 20 days.

    Parameters
    returns : pd.Series
        Log-return calculated in compute_log_returns
    window : int
        Rolling window length in number of days.

    Returns
    pd.Series
        volatility calculated in window.
    """
    if window < 2:                                 #check to make sure there is at least 2 entries to calculate
        raise ValueError("window must be >= 2")

    #use baked in pandas.rolling function to calculate standard deviation of entries in teh window
    to_return = returns.rolling(window=window, min_periods=window).std()    
    return to_return


def label_regime(returns, window = 20, threshold = None):
    """
    Classify each period as 'calm' or 'volatile' based on  volatility in rolling widnow

    A window of log-returns is compared to a threshold. if it is greater than the threshold it
    is classified as volatile, if it is below it is calm. By default the median of the rolling
    window volitility is used as the threshold. this allows for the threshold to be relative to 
    each stock by default, but leads to splitting into nearly equal sizes instead of truly finding
    calm and volatile.

    Parameters
    returns : pd.Series
        Log-return calculated in compute_log_returns
    window : int
        Rolling window length in number of days.
    threshold : float
        sets the threshold where window will be considered volatile

    Returns
    pd.Series
        column labels of either calm or volatile 
    """

    vol = rolling_volatility(returns, window=window)   #get the rolling volatility column
    vol = vol.dropna()                                 #clean out any NaN entries
    if len(vol) == 0:                                  #make sure the rolling volatility window is not empty
        raise ValueError("unable to compute rolling volatility window")

    if threshold is None:                       #if the threshold is in default state, reset it to be the median
        threshold = vol.median()

    #use a lambda function and pandas.apply to apply labels to the labels series
    labels = vol.apply(lambda v: "volatile" if v > threshold else "calm") 
    labels = pd.Series(labels, name="regime")      #add a header column to the series 
    return labels





if __name__ == "__main__":    #allow module to run on its own to test
    sample_csv = Path(__file__).parent / "data" / "aapl_sample.csv"
    prices = load_daily_data(sample_csv)
    rets = compute_log_returns(prices)
    regimes = label_regime(rets, window=20)

    print(f"Computed {len(rets)} log-returns")
    print("Regime counts:")
    print(regimes.value_counts())
