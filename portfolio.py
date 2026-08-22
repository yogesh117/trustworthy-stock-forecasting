# Team 2
# Olin Dsouza (odsouza1@stevens.edu), Yogesh Patil (yp36032@stevens.edu), Kevin Gwinn (kgwinn@stevens.edu)
# Date: 8/16/26
# Description: Engineering Python Final Project, Trustworthy Stock Forecaster
# File name: portfolio.py

"""
portfolio.py module defines the class StockAssets which has a stock symbols cleaned and calculated
data. It also uses composition to create class Portfolio which has StockAssets. This builds a
simple version of a "stock portfolio" that can be evaluated for returns.

portfolio.py also implements the dunders of __str__ and __len__ to report user friendly versions
of printing the portfolio and reporting the number of stock assets in the portfolio.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from data_loader import compute_log_returns, load_daily_data


class StockAssets:
    """
    Class creates an instance of a single stock symbol's historical data.

    Attributes
    symbol (str)
        The stock symbol for the asset.
    prices (DataFrame)
        Asset's data formatted and verified in a dataframe, created by data_loader.load_daily_data()
    returns : (Dataframe)
        Return data calculated by data_loader.compute_log_returs(). 
    """

    def __init__(self, symbol, prices, returns=None):    #initializes asset with symbol, prices, and returns

        self.symbol = symbol
        self.prices = prices

        if returns is None:                             #returns should be passed in, if they are not they need to be calculated
            returns = compute_log_returns(self.prices)
        self.returns = returns

    @classmethod
    def from_csv(cls, symbol, csv_path):
        """
        Alternative constructor to create an asset directly from csv file using the 
        built in classmethod decorator.

        #note: this code was originally found online at the following source and adapted for this project.
        It uses decorators as taught in class, but the specific csv function was found during research and used.
        https://stackoverflow.com/questions/39847154/pythonic-superclass-with-classmethod-constructor-override-inherit
        
        """
        prices = load_daily_data(csv_path)
        return cls(symbol, prices)

    def __len__(self):
        """
        returns len(self.prices) as an easy way to check the number of days in a stocks asset
        """
        return len(self.prices)

    def __str__(self):
        """
        overload print function to print the stock symbol and number of days of data
        """
        return f"StockAssets({self.symbol}, n_days={len(self)})"


class Portfolio:
    """
    Class creates a collection of stock assets that can be used to calculate and process 
    returns on a set of stock assets that exist within the portfolio.
    """

    def __init__(self, name="Portfolio"):     #create a portfolio instance with name and empty assets set
        self.name = name
        self._assets = {}

    def add_asset(self, asset):
        """
        Add an asset to the portfolio
        parameters
            asset (str) stock symbol to add

        returns
            Portfolio object with added asset
        """
        if not isinstance(asset, StockAssets):    #check that the asset being added is an object of class StockAssets
            raise TypeError("Portfolio only accepts StockAssets objects")    #raise exception if asset is not correct class
        self._assets[asset.symbol] = asset    #add stock data with symbol as key
        return self

    def get_asset(self, symbol):
        """
        getter for stock asset, takes a symbol and returns data tied to that key
        parameter
            symbol(str)
            stock symbol for data request
        returns (StockAssets)
            returns StockAssets object that is connected to the symbol's key
        """
        if symbol not in self._assets:    #check that the symbol is in the portfolio
            raise KeyError(f"No asset '{symbol}' in portfolio '{self.name}'")    #raise exception if not present
        return self._assets[symbol]

    def symbols(self):
        """
        returns a list of assets in a portfolio (stock symbols only, not the data)
        """
        return list(self._assets.keys())

    def assets(self):
        """
        returns the values of all stocks in a portfolio (not the keys)
        """
        return iter(self._assets.values())

    # -- summary statistics -----------------------------------------------
    def summary(self):
        """
        implements dictionary comprehension to create and return a dict of a portfolio. summarizing
        the number of days, average return, and volume for each symbol in a portfolio.

        """
        return {
            symbol: {
                "n_days": len(asset),      #keys n_days to the number of days of data
                "mean_return": float(asset.returns.mean()),   #calculates the average of returns and assigns to key mean_return
                "vol": float(asset.returns.std()),   #calculate the standard deviation of volume of a stock - used later in regime calculations
            }
            for symbol, asset in self._assets.items()  
        }

    def __len__(self):
        """
        Overload dunder to make get the size of a portfolio (number of assets) easier
        """
        return len(self._assets)

    def __str__(self):
        """
        Overload function for printing a portfolio easily. Prints the portfolio name, number of assets, then the
        symbol of each asset, number of days of data, the average return, and the standard deviation of the volume.
        """
        lines = [f"Portfolio '{self.name}' ({len(self)} assets):"]      #create a list and label the portfolio name and size
        for symbol, stats in self.summary().items():    #iterate through each symbol and pull data from the summary
            lines.append(                               #add the symbol and data to the lines list
                f"{symbol}: {stats['n_days']} days, "
                f"mean_return={stats['mean_return']:.6f}, "
                f"vol={stats['vol']:.6f}"
            )
        #__str__ must return a str, and it needs to be formatted nicely. this section converts from list to str
        list_to_str = ""            #create an empty string to be added to
        for l in lines:             #iterate through the list
            list_to_str += l        #add each line to our string
            list_to_str += "\n"     #put a newline at the end of each line in the string
        list_to_str = list_to_str.rstrip()  #after the loop strip out the trailing newline
        return list_to_str          #return the new long string that has all the list data.


if __name__ == "__main__":
    # create stock assets directly from csv files
    aapl = StockAssets.from_csv("AAPL", Path(__file__).parent / "data" / "aapl_sample.csv")
    msft = StockAssets.from_csv("MSFT", Path(__file__).parent / "data" / "msft_sample.csv")
    jpm = StockAssets.from_csv("JPM", Path(__file__).parent / "data" / "jpm_sample.csv")

    #create a portfolio then add each stock asset to the portfolio
    pf = Portfolio(name="Demo Portfolio")
    pf.add_asset(aapl)
    pf.add_asset(msft)
    pf.add_asset(jpm)

    #print the portfolio
    print(pf)
    