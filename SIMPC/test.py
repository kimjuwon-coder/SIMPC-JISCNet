'''
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import torch
torch.set_default_dtype(torch.float)
df = pdr.get_data_yahoo('^GSPC', start='1980-01-01', end='2020-01-01').reset_index()
print(df)
'''
from utils.load_data import *
from models.model_params import *
from models.pattern_recognition_module import *
from models.train_ftsdiffusion import *

import yfinance as yf
#df = yf.download('^GSPC', start="2023-05-01", end="2023-12-31")
#df['Close'].to_csv('data\data_toy_l10-20' + '_timeseries.csv', index=False)
fts = get_fts(ticker='^GSPC', start_date='1980-01-01',  end_date='2020-01-01')
#print('fts: ', fts)
centroids, labels, subsequences, segmentation = train_ftsdiffusion_recognition(fts, store_model=True)
print('centroids: ', centroids)
print('labels: ', labels)
print('subsequences: ', subsequences)
print('segmentation: ', segmentation)