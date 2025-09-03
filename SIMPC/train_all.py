#############################################
# Important Note                            #
# Due to the inherentrandomness, suggest to #
# train and select the modules separately   #
#############################################



from utils.load_data import *
from models.model_params import *
from models.pattern_recognition_module import *
from models.train_ftsdiffusion import *

 

def train_all():
  # from models.model_params
  # Hyperparameters of pattern recognition module
  dataname = prm_params['dataname'] # 'sp500'
  # Get the historical time series (S&P 500 as example)
  # from utils.load_data
  # yfinance에서 financial time series (종가)정보를 얻음
  # 종가 정보가 나열된 시계열 데이터
  fts = get_fts(ticker='^GSPC', fts_name=dataname, start_date='1980-01-01',  end_date='2020-01-01')
  # Train the modules
  # from models.train_ftsdiffusion
  train_ftsdiffusion(fts)









