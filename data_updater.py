
### Early updater via an API. Use with caution.  DO NOT CHANGE API KEY

import urllib.request
import csv
import io
import pandas as pd
from pathlib import Path

API_KEY = 'OGLS55Q6YJLUVQPP'     #API key, do not change

symbol = 'IBM'
  
function_call = 'TIME_SERIES_DAILY'
url = f'https://www.alphavantage.co/query?function={function_call}Y&symbol={symbol}&outputsize=full&datatype=csv&apikey={API_KEY}'

output = Path(__file__).parent / "data" / f"{symbol}.csv"

def updater():
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode("utf-8")
        #print(data)

        with open(output, "w", encoding="utf-8") as file:
            for line in [data]:
                cleaned_line = line.rstrip("\n")
                print(cleaned_line, file=file)
            

 
    except:
        print("something went wrong, no file created")

if __name__ == "__main__":    #allow module to run on its own to test
    updater()