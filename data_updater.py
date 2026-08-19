
### Early updater via an API. Use with caution.  DO NOT CHANGE API KEY
### Kevin will add docstrings and comments

import urllib.request
import os
import pandas as pd
from pathlib import Path

API_KEY = 'OGLS55Q6YJLUVQPP'     #API key, do not change

def cleanup_symbols():  #remove files with under 10 rows (that means the file does not have enough data, probably because an API fail)
    target_dir = Path(__file__).parent / "data"
    for f in target_dir.iterdir():
        if f.is_file():
            df = pd.read_csv(f)

            row_count = len(df)
            #print(f"file {f} - Total rows: {row_count}")
            if row_count < 10:
                 if os.path.exists(f):
                    os.remove(f)
                 

def add_update_symbol(symbol='AAPL'):
    function_call = 'TIME_SERIES_DAILY'
    url = f'https://www.alphavantage.co/query?function={function_call}&symbol={symbol}&datatype=csv&outputsize=full&apikey={API_KEY}'
    temp = Path(__file__).parent / "data" / f"temp.csv"
    output = Path(__file__).parent / "data" / f"{symbol}.csv"
    try:
        if os.path.exists(temp):
            os.remove(temp)

        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
        #print([data][0])

        with open(temp, "w", encoding="utf-8") as file:
            for line in [data]:
                cleaned_line = line.rstrip("\n")
                if len(cleaned_line) != 0:
                    print(cleaned_line, file=file)
            
        df = pd.read_csv(temp)
        df = df.head(1000)
        df = df.rename(columns={"timestamp": "Date", "high":"High","low":"Low","open":"Open","close":"Close","volume":"Volume"})

       # print(len(df))
        if len(df) > 200:
            df.to_csv(output, index=False)

        if os.path.exists(temp):
            os.remove(temp)
 
    except:
        print("something went wrong, no file created")

def remove_symbol(symbol):
    target = Path(__file__).parent / "data" / f"{symbol}.csv"
    if symbol not in ["aapl_sample", "jpm_sample","msft_sample"]:
        if os.path.exists(target):
            os.remove(target)

if __name__ == "__main__":    #allow module to run on its own to test
    add_update_symbol('AAPL')
    #cleanup_symbols()
    #remove_symbol("AAPL")