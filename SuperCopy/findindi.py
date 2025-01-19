# calculate each individual indicator

import pandas as pd
class PSAR:
  

  def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
    self.max_af = max_af
    self.init_af = init_af
    self.af = init_af
    self.af_step = af_step
    self.extreme_point = None
    self.high_price_trend = []
    self.low_price_trend = []
    self.high_price_window = deque(maxlen=2)
    self.low_price_window = deque(maxlen=2)

    self.psar_list = []
    self.af_list = []
    self.ep_list = []
    self.high_list = []
    self.low_list = []
    self.trend_list = []
    self._num_days = 0

  def calcPSAR(self, high, low):
    if self._num_days >= 3:
      psar = self._calcPSAR()
    else:
      psar = self._initPSARVals(high, low)

    psar = self._updateCurrentVals(psar, high, low)
    self._num_days += 1

    return psar

  def _initPSARVals(self, high, low):
    if len(self.low_price_window) <= 1:
      self.trend = None
      self.extreme_point = high
      return None

    if self.high_price_window[0] < self.high_price_window[1]:
      self.trend = 1
      psar = min(self.low_price_window)
      self.extreme_point = max(self.high_price_window)
    else: 
      self.trend = 0
      psar = max(self.high_price_window)
      self.extreme_point = min(self.low_price_window)

    return psar

  def _calcPSAR(self):
    prev_psar = self.psar_list[-1]
    if self.trend == 1: # Up
      psar = prev_psar + self.af * (self.extreme_point - prev_psar)
      psar = min(psar, min(self.low_price_window))
    else:
      psar = prev_psar - self.af * (prev_psar - self.extreme_point)
      psar = max(psar, max(self.high_price_window))

    return psar

  def _updateCurrentVals(self, psar, high, low):
    if self.trend == 1:
      self.high_price_trend.append(high)
    elif self.trend == 0:
      self.low_price_trend.append(low)

    psar = self._trendReversal(psar, high, low)

    self.psar_list.append(psar)
    self.af_list.append(self.af)
    self.ep_list.append(self.extreme_point)
    self.high_list.append(high)
    self.low_list.append(low)
    self.high_price_window.append(high)
    self.low_price_window.append(low)
    self.trend_list.append(self.trend)

    return psar

  def _trendReversal(self, psar, high, low):
    # Checks for reversals
    reversal = False
    if self.trend == 1 and psar > low:
      self.trend = 0
      psar = max(self.high_price_trend)
      self.extreme_point = low
      reversal = True
    elif self.trend == 0 and psar < high:
      self.trend = 1
      psar = min(self.low_price_trend)
      self.extreme_point = high
      reversal = True

    if reversal:
      self.af = self.init_af
      self.high_price_trend.clear()
      self.low_price_trend.clear()
    else:
        if high > self.extreme_point and self.trend == 1:
          self.af = min(self.af + self.af_step, self.max_af)
          self.extreme_point = high
        elif low < self.extreme_point and self.trend == 0:
          self.af = min(self.af + self.af_step, self.max_af)
          self.extreme_point = low

    return psar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import talib as ta
from scipy.signal import argrelextrema
from collections import deque

def rsiuse(dfa1):
    global data,hh,lh,ll,hl,hh_idx,lh_idx,ll_idx,hl_idx,price,dates
    data=dfa1
    price = data['Close'].values
    dates = data.index
    # Get higher highs, lower lows, etc.
    order = 5
    hh = getHigherHighs(price, order)
    lh = getLowerHighs(price, order)
    ll = getLowerLows(price, order)
    hl = getHigherLows(price, order)
    # Get confirmation indices
    hh_idx = np.array([i[1] + order for i in hh])
    lh_idx = np.array([i[1] + order for i in lh])
    ll_idx = np.array([i[1] + order for i in ll])
    hl_idx = np.array([i[1] + order for i in hl])
    return data
def getHigherLows(data: np.array, order=5, K=2):
  # Get lows
  low_idx = argrelextrema(data, np.less, order=order)[0]
  lows = data[low_idx]
  # Ensure consecutive lows are higher than previous lows
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(low_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if lows[i] < lows[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getLowerHighs(data: np.array, order=5, K=2):
  # Get highs
  high_idx = argrelextrema(data, np.greater, order=order)[0]
  highs = data[high_idx]
  # Ensure consecutive highs are lower than previous highs
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(high_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if highs[i] > highs[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getHigherHighs(data: np.array, order=5, K=2):
  # Get highs
  high_idx = argrelextrema(data, np.greater, order=order)[0]
  highs = data[high_idx]
  # Ensure consecutive highs are higher than previous highs
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(high_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if highs[i] < highs[i-1]:
      ex_deque.clear()
    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())
  
  return extrema

def getLowerLows(data: np.array, order=5, K=2):
  # Get lows
  low_idx = argrelextrema(data, np.less, order=order)[0]
  lows = data[low_idx]
  # Ensure consecutive lows are lower than previous lows
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(low_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if lows[i] > lows[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def calcRSI(data, P=14):
  data['diff_close'] = data['Close'] - data['Close'].shift(1)
  data['gain'] = np.where(data['diff_close']>0, data['diff_close'], 0)
  data['loss'] = np.where(data['diff_close']<0, np.abs(data['diff_close']), 0)
  data[['init_avg_gain', 'init_avg_loss']] = data[
    ['gain', 'loss']].rolling(P).mean()
  avg_gain = np.zeros(len(data))
  avg_loss = np.zeros(len(data))
  for i, _row in enumerate(data.iterrows()):
    row = _row[1]
    if i < P - 1:
      last_row = row.copy()
      continue
    elif i == P-1:
      avg_gain[i] += row['init_avg_gain']
      avg_loss[i] += row['init_avg_loss']
    else:
      avg_gain[i] += ((P - 1) * avg_gain[i-1] + row['gain']) / P
      avg_loss[i] += ((P - 1) * avg_loss[i-1] + row['loss']) / P
          
    last_row = row.copy()
      
  data['avg_gain'] = avg_gain
  data['avg_loss'] = avg_loss
  data['RS'] = data['avg_gain'] / data['avg_loss']
  data['RSI'] = 100 - 100 / (1 + data['RS'])
  return data
def getHHIndex(data: np.array, order=5, K=2):
  extrema = getHigherHighs(data, order, K)
  idx = np.array([i[-1] + order for i in extrema])
  return idx[np.where(idx<len(data))]

def getLHIndex(data: np.array, order=5, K=2):
  extrema = getLowerHighs(data, order, K)
  idx = np.array([i[-1] + order for i in extrema])
  return idx[np.where(idx<len(data))]

def getLLIndex(data: np.array, order=5, K=2):
  extrema = getLowerLows(data, order, K)
  idx = np.array([i[-1] + order for i in extrema])
  return idx[np.where(idx<len(data))]

def getHLIndex(data: np.array, order=5, K=2):
  extrema = getHigherLows(data, order, K)
  idx = np.array([i[-1] + order for i in extrema])
  return idx[np.where(idx<len(data))]

def getPeaks(data, key='Close', order=5, K=2):
  vals = data[key].values
  hh_idx = getHHIndex(vals, order, K)
  lh_idx = getLHIndex(vals, order, K)
  ll_idx = getLLIndex(vals, order, K)
  hl_idx = getHLIndex(vals, order, K)

  data[f'{key}_highs'] = np.nan
  data[f'{key}_highs'][hh_idx] = 1
  data[f'{key}_highs'][lh_idx] = -1
  data[f'{key}_highs'] = data[f'{key}_highs'].ffill().fillna(0)
  data[f'{key}_lows'] = np.nan
  data[f'{key}_lows'][ll_idx] = 1
  data[f'{key}_lows'][hl_idx] = -1
  data[f'{key}_lows'] = data[f'{key}_highs'].ffill().fillna(0)
  return data
def RSIDivergenceStrategy(data, P=14, order=5, K=2):
  data = getPeaks(data, key='Close', order=order, K=K)
  data = calcRSI(data, P=P)
  data = getPeaks(data, key='RSI', order=order, K=K)

  position = np.zeros(data.shape[0])
  for i, (t, row) in enumerate(data.iterrows()):
    if np.isnan(row['RSI']):
        continue
    # If no position is on
    if position[i-1] == 0:
      # Buy if indicator to higher low and price to lower low
      if row['Close_lows'] == -1 and row['RSI_lows'] == 1:
        if row['RSI'] < 50:
          position[i] = 1
          entry_rsi = row['RSI'].copy()

      # Short if price to higher high and indicator to lower high
      elif row['Close_highs'] == 1 and row['RSI_highs'] == -1:
        if row['RSI'] > 50:
          position[i] = -1
          entry_rsi = row['RSI'].copy()

    # If current position is long
    elif position[i-1] == 1:
        if row['RSI'] < 50 and row['RSI'] < entry_rsi:
            position[i] = 1
  
    # If current position is short
    elif position[i-1] == -1:
        if row['RSI'] < 50 and row['RSI'] > entry_rsi:
            position[i] = -1

  data['position'] = position
  return calcReturns(data)

def calcReturns(df):
  # Helper function to avoid repeating too much code
  df['returns'] = df['Close'] / df['Close'].shift(1)
  df['log_returns'] = np.log(df['returns'])
  df['strat_returns'] = df['position'].shift(1) * df['returns']
  df['strat_log_returns'] = df['position'].shift(1) * df['log_returns']
  df['cum_returns'] = np.exp(df['log_returns'].cumsum()) - 1
  df['strat_cum_returns'] = np.exp(df['strat_log_returns'].cumsum()) - 1
  df['peak'] = df['cum_returns'].cummax()
  df['strat_peak'] = df['strat_cum_returns'].cummax()
  return df
def rsidiver():
    position_df = RSIDivergenceStrategy(data)
    positions = position_df['position'].tolist()
    rsi_divergent_signal=positions
    return rsi_divergent_signal
def findindis(Name):
    import datetime
    from datetime import date
    from datetime import timedelta
    import pandas_datareader.data as web
    import matplotlib.pyplot as plt
    import numpy as np
    import talib as ta
    import pandas as pd
    import yfinance as yf
    from collections import deque
    import pandas_datareader as pdr
    from scipy import stats
    from scipy import signal as sg
    import plotly.express as px
    import plotly.graph_objects as go
    global obv
	#preprocess data
    yesterday = date.today() - timedelta(hours = 12 )
    yesterday=str(yesterday)
    
    Eyear = int(yesterday[:4])
    Emonth = int(yesterday[5:7])
    Eday = int(yesterday[8:10])
    Syear=2022
    Smonth=1
    Sday=2
    Sdate ="%s-%s-%s" % (Syear,Smonth,Sday)
    Edate ="%s-%s-%s" % (Eyear,Emonth,Eday)
    start=datetime.datetime(Syear,Smonth,Sday)
    end=datetime.datetime(Eyear,Emonth,Eday)
    Stockdata=yf.download(Name,start,end)
    dfa1=Stockdata
    Byear = Syear-1
    start=datetime.datetime(Byear,Smonth,Sday)
    end=datetime.datetime(Eyear,Emonth,Eday)
    Stockdata=yf.download(Name,start,end)
    dfa1_all=Stockdata

    def psar_changetrend_signal_func():
        global psar_changetrend_signal,psarc
        ticker = Name
        yfObj = yf.Ticker(ticker)
        yfO = yfObj.history(start=Sdate, end=Edate)[['High', 'Low']].round(2)
        yfO=yfO.loc[Sdate:Edate]
        yfO.reset_index(inplace=True)
        indic = PSAR()
        yfO['PSAR'] = yfO.apply(lambda x: indic.calcPSAR(x['High'], x['Low']), axis=1)
        psarc=yfO['PSAR']
        yfO['Trend'] = indic.trend_list
        indic._calcPSAR()
        psar_changetrend_signal=[0]*len(yfO['Trend'])
        for idx,i in enumerate(yfO['Trend']):
            if (pd.isna(i) or idx==0):
                psar_changetrend_signal[idx]=0
            else:
                if (yfO['Trend'][idx-1]==0 and yfO['Trend'][idx]==1):
                    psar_changetrend_signal[idx]=1
                elif(yfO['Trend'][idx-1]==1 and yfO['Trend'][idx]==0):
                    psar_changetrend_signal[idx]=-1
                else:
                    psar_changetrend_signal[idx]=0
    psar_changetrend_signal_func()
    
    def psar_trend_signal_func():
        global psar_trend_signal
        ticker = Name
        yfObj = yf.Ticker(ticker)
        yfO = yfObj.history(start=Sdate, end=Edate)[['High', 'Low']].round(2)
        yfO=yfO.loc[Sdate:Edate]
        yfO.reset_index(inplace=True)
        indic = PSAR()
        yfO['PSAR'] = yfO.apply(lambda x: indic.calcPSAR(x['High'], x['Low']), axis=1)
        yfO['Trend'] = indic.trend_list
        indic._calcPSAR()
        psar_trend_signal=[0]*len(yfO['Trend'])
        for idx,i in enumerate(yfO['Trend']):
            if (yfO['Trend'][idx]==1):
                psar_trend_signal[idx]=1
            elif(yfO['Trend'][idx]==0):
                psar_trend_signal[idx]=-1
            else:
                psar_trend_signal[idx]=0
    psar_trend_signal_func()
    global MA89,MA100,atr
    MA30=ta.MA(dfa1_all['Close'], timeperiod=30, matype=0)
    MA30=MA30.loc[Sdate:Edate]
    MA30=MA30.array
    dfa1['MA30']=MA30

    MA50=ta.MA(dfa1_all['Close'], timeperiod=50, matype=0)
    MA50=MA50.loc[Sdate:Edate]
    MA50=MA50.array
    dfa1['MA50']=MA50

    MA200=ta.MA(dfa1_all['Close'], timeperiod=200, matype=0)
    MA200=MA200.loc[Sdate:Edate]
    MA200=MA200.array
    dfa1['MA200']=MA200
    
    MA100=ta.MA(dfa1_all['Close'], timeperiod=100, matype=0)
    MA100=MA100.loc[Sdate:Edate]
    MA100=MA100.array
    dfa1['MA100']=MA100
      
    MA25=ta.MA(dfa1_all['Close'], timeperiod=25, matype=0)
    MA25=MA25.loc[Sdate:Edate]
    MA25=MA25.array
    dfa1['MA25']=MA25
    
    MA89=ta.MA(dfa1_all['Close'], timeperiod=89, matype=0)
    MA89=MA89.loc[Sdate:Edate]
    MA89=MA89.array
    dfa1['MA89']=MA89
    
    MA21=ta.MA(dfa1_all['Close'], timeperiod=21, matype=0)
    MA21=MA21.loc[Sdate:Edate]
    MA21=MA21.array
    dfa1['MA21']=MA21
    global ADX
    ADX=ta.ADX(dfa1_all['High'], dfa1_all['Low'], dfa1_all['Close'], timeperiod=14)
    ADX=ADX.loc[Sdate:Edate]
    ADX=ADX.array
    dfa1['ADX']=ADX
    plus_DMI = ta.PLUS_DM(dfa1_all['High'], dfa1_all['Low'], timeperiod=14)
    minus_DMI= ta.MINUS_DM(dfa1_all['High'], dfa1_all['Low'], timeperiod=14)
    plus_DMI=plus_DMI.loc[Sdate:Edate]
    minus_DMI=minus_DMI.loc[Sdate:Edate]
    plus_DMI=plus_DMI.array
    minus_DMI=minus_DMI.array
    dfa1['plus_DMI']=plus_DMI
    dfa1['minus_DMI']=minus_DMI
    prices, pdi, ndi, adx=dfa1['Close'], dfa1['plus_DMI'], dfa1['minus_DMI'], dfa1['ADX']

    MA5=ta.MA(dfa1_all['Close'], timeperiod=5, matype=0)
    MA5=MA5.loc[Sdate:Edate]
    MA5=MA5.array
    dfa1['MA5']=MA5

    MA10=ta.MA(dfa1_all['Close'], timeperiod=10, matype=0)
    MA10=MA10.loc[Sdate:Edate]
    MA10=MA10.array
    dfa1['MA10']=MA10   

    obv = ta.OBV(dfa1_all["Close"], dfa1_all["Volume"])
    obv_mean=obv.rolling(20).mean()
    obv_mean=obv_mean.loc[Sdate:Edate]
    data=dfa1['Open']

    def ma25_89_signal_func():
        global ma25_89_signal
        signal=0
        ma25_89_signal=[]
        for i in range(len(dfa1)):
            if dfa1['MA25'][i] > dfa1['MA89'][i]:
                if signal != 1:
                    signal = 1
                    ma25_89_signal.append(signal)
                else:
                    ma25_89_signal.append(0)
            elif dfa1['MA25'][i] < dfa1['MA89'][i]:
                if signal != -1:
                    signal = -1
                    ma25_89_signal.append(signal)
                else:
                    ma25_89_signal.append(0)
            else:
                ma25_89_signal.append(0)

    def ma21_89_signal_func():
        global ma21_89_signal
        signal=0
        ma21_89_signal=[]
        for i in range(len(dfa1)):
            if dfa1['MA21'][i] > dfa1['MA89'][i]:
                if signal != 1:
                    signal = 1
                    ma21_89_signal.append(signal)
                else:
                    ma21_89_signal.append(0)
            elif dfa1['MA21'][i] < dfa1['MA89'][i]:
                if signal != -1:
                    signal = -1
                    ma21_89_signal.append(signal)
                else:
                    ma21_89_signal.append(0)
            else:
                ma21_89_signal.append(0)
    def ma30_200_signal_func():
        global ma30_200_signal
        ma30_200_signal = []
        signal = 0
        for i in range(len(dfa1)):
            if dfa1['MA30'][i] > dfa1['MA200'][i]:
                if signal != 1:
                    signal = 1
                    ma30_200_signal.append(signal)
                else:
                    ma30_200_signal.append(0)
            elif dfa1['MA30'][i] < dfa1['MA200'][i]:
                if signal != -1:
                    signal = -1
                    ma30_200_signal.append(signal)
                else:
                    ma30_200_signal.append(0)
            else:
                ma30_200_signal.append(0)
                     
    def ma5_10_signal_func():
        global ma5_10_signal
        signal = 0
        ma5_10_signal=[]
        for i in range(len(dfa1)):
            if dfa1['MA5'][i] > dfa1['MA10'][i]:
                if signal != 1:
                    signal = 1
                    ma5_10_signal.append(signal)
                else:
                    ma5_10_signal.append(0)
            elif dfa1['MA5'][i] < dfa1['MA10'][i]:
                if signal != -1:
                    signal = -1
                    ma5_10_signal.append(signal)
                else:
                    ma5_10_signal.append(0)
            else:
                ma5_10_signal.append(0)

    def ma50_200_signal_func():
        global ma50_200_signal
        ma50_200_signal=[]
        signal = 0
        for i in range(len(dfa1)):
            if dfa1['MA50'][i] > dfa1['MA200'][i]:
                if signal != 1:
                    signal = 1
                    ma50_200_signal.append(signal)
                else:
                    ma50_200_signal.append(0)
            elif dfa1['MA50'][i] < dfa1['MA200'][i]:
                if signal != -1:
                    signal = -1
                    ma50_200_signal.append(signal)
                else:
                    ma50_200_signal.append(0)
            else:
                ma50_200_signal.append(0) 
    
    def ma50_100_signal_func():
        global ma50_100_signal
        ma50_100_signal=[]
        signal=0
        for i in range(len(dfa1)):
            if dfa1['MA50'][i] > dfa1['MA100'][i]:
                if signal != 1:
                    signal = 1
                    ma50_100_signal.append(signal)
                else:
                    ma50_100_signal.append(0)
            elif dfa1['MA50'][i] < dfa1['MA100'][i]:
                if signal != -1:
                    signal = -1
                    ma50_100_signal.append(signal)
                else:
                    ma50_100_signal.append(0)
            else:
                ma50_100_signal.append(0)  

    def ma30_50_signal_func():
        global ma30_50_signal
        ma30_50_signal = []
        signal = 0        
        for i in range(len(dfa1)):
            if dfa1['MA30'][i] > dfa1['MA50'][i]:
                if signal != 1:
                    signal = 1
                    ma30_50_signal.append(signal)
                else:
                    ma30_50_signal.append(0)
            elif dfa1['MA30'][i] < dfa1['MA50'][i]:
                if signal != -1:
                    signal = -1
                    ma30_50_signal.append(signal)
                else:
                    ma30_50_signal.append(0)
            else:
                ma30_50_signal.append(0)

    def obv_signal_func():
        global obv_signal
        obv_signal = []
        signal = 0
        for i in range(len(obv_mean)):
            if obv[i] > obv_mean[i]:
                if signal != 1:
                    signal = 1
                    obv_signal.append(signal)
                else:
                    obv_signal.append(0)
            elif obv[i] < obv_mean[i]:
                if signal != -1:
                    signal = -1
                    obv_signal.append(signal)
                else:
                    obv_signal.append(0)
            else:
                obv_signal.append(0)
    
    def sma100_signal_func():
        global sma100_signal,prices,MA100
        MA100=ta.MA(dfa1_all['Close'], timeperiod=100, matype=0)
        MA100=MA100.loc[Sdate:Edate]
        MA100=MA100.array
        dfa1['MA100']=MA100
        prices=dfa1['Close']
        signal = 0
        sma100_signal=[]
        for i in range(len(prices)):
            if i==0 :
                signal=0
            else:
                if prices[i-1]<=MA100[i] and prices[i]>=MA100[i]:
                    signal = 1
                elif prices[i-1]>=MA100[i] and prices[i]<=MA100[i]:
                    signal=-1
                else:
                    signal=0
            sma100_signal.append(signal)

    def sma89_signal_func():
        global sma89_signal,MA89
        MA89=ta.MA(dfa1_all['Close'], timeperiod=89, matype=0)
        MA89=MA89.loc[Sdate:Edate]
        MA89=MA89.array
        dfa1['MA89']=MA89
        prices=dfa1['Close']
        signal = 0
        sma89_signal=[]
        for i in range(len(prices)):
            if (i==0):
                signal=0
            else:
                if prices[i-1]<=MA89[i] and prices[i]>=MA89[i]:
                    signal = 1
                elif prices[i-1]>=MA89[i] and prices[i]<=MA89[i]:
                    signal=-1
                else:
                    signal=0
            sma89_signal.append(signal)

    def ma30_100_signal_func():
        global ma30_100_signal
        signal = 0
        ma30_100_signal=[]        
        for i in range(len(dfa1)):
            if dfa1['MA30'][i] > dfa1['MA100'][i]:
                if signal != 1:
                    signal = 1
                    ma30_100_signal.append(signal)
                else:
                    ma30_100_signal.append(0)
            elif dfa1['MA30'][i] < dfa1['MA100'][i]:
                if signal != -1:
                    signal = -1
                    ma30_100_signal.append(signal)
                else:   
                    ma30_100_signal.append(0)
            else:
                ma30_100_signal.append(0)

    def macd_crossover_signal_func():
        global macd_crossover_signal,MACD
        macd, macdsignal, macdhist = ta.MACD(dfa1_all['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        MACD=macd.loc[Sdate:Edate]
        dfa1['MACD']=MACD
        dfa1['macdsignal']=macdsignal
        dfa1['macdhist']=macdhist
        data_macd=dfa1
        data=dfa1
        prices=dfa1['Open']
        macd_crossover_signal = []
        signal = 0
        for i in range(len(data)):
            if data_macd['MACD'][i] > data_macd['macdsignal'][i]:
                if signal != 1:
                    signal = 1
                    macd_crossover_signal.append(signal)
                else:
                    macd_crossover_signal.append(0)
            elif data_macd['MACD'][i] < data_macd['macdsignal'][i]:
                if signal != -1:
                    signal = -1
                    macd_crossover_signal.append(signal)
                else:
                    macd_crossover_signal.append(0)
            else:
                macd_crossover_signal.append(0)

    def macd_overzero_signal_func():
        global macd_overzero_signal
        macd, macdsignal, macdhist = ta.MACD(dfa1_all['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        MACD=macd.loc[Sdate:Edate]
        dfa1['MACD']=MACD
        dfa1['macdsignal']=macdsignal
        dfa1['macdhist']=macdhist
        data_macd=dfa1
        data=dfa1
        prices=dfa1['Open']
        macd_overzero_signal=[]
        for i in range(len(data)):
            if data_macd['MACD'][i] > 0:
                macd_overzero_signal.append(1)
            else:
                macd_overzero_signal.append(-1)

    def macd_kongpop_signal_func():
        global macd_kongpop_signal
        macd, macdsignal, macdhist = ta.MACD(dfa1_all['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        MACD=macd.loc[Sdate:Edate]
        dfa1['MACD']=MACD
        dfa1['macdsignal']=macdsignal
        dfa1['macdhist']=macdhist
        data_macd=dfa1
        data=dfa1
        prices=dfa1['Open']
        macd_kongpop_signal=[]
        for i in range(len(data)):
            if i==0:
                pass
            if data_macd['MACD'][i-1] <= -0.3 and data_macd['MACD'][i-1] >=-0.3:
                macd_kongpop_signal.append(1)
            elif data_macd['MACD'][i-1] >= 0.5 and data_macd['MACD'][i-1] <=0.5:
                macd_kongpop_signal.append(-1)
            else:
                macd_kongpop_signal.append(0)

    def adx_signal_func():
        global adx_signal
        adx_signal = []
        signal = 0
        for i in range(len(prices)):
            if adx[i-1] < 25 and adx[i] > 25 and pdi[i] > ndi[i]:
                if signal != 1:
                    signal = 1
                    adx_signal.append(signal)
                else:
                    adx_signal.append(0)
            elif adx[i-1] < 25 and adx[i] > 25 and ndi[i] > pdi[i]:
                if signal != -1:
                    signal = -1
                    adx_signal.append(signal)
                else:   
                    adx_signal.append(0)
            else:
                adx_signal.append(0)
    
    fastk, fastd = ta.STOCHRSI(dfa1_all['Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    K=fastk.loc[Sdate:Edate]
    K=K.array
    dfa1['%K']=K
    D=fastd.loc[Sdate:Edate]
    D=D.array
    dfa1['%D']=D
    k=dfa1['%K']
    d=dfa1['%D']

    def stoch_signal_func():
        global stoch_signal,k
        stoch_signal = []
        signal = 0
        for i in range(len(dfa1['Close'])):
            k=dfa1['%K']
            d=dfa1['%D']
            if k[i] < 20 and d[i] < 20 and k[i] < d[i]:
                if signal != 1:
                    signal = 1
                    stoch_signal.append(signal)
                else:
                    stoch_signal.append(0)
            elif k[i] > 80 and d[i] > 80 and k[i] > d[i]:
                if signal != -1:
                    signal = -1
                    stoch_signal.append(signal)
                else:
                    stoch_signal.append(0)
            else:
                stoch_signal.append(0)

    def adx_sto_kp_signal_func():
        global adx_sto_kp_signal
        adx_sto_kp_signal = []
        signal = 0
        zone_sto = 0
        for i in range(len(prices)):
            if d[i]>=80:
                zone_sto=1
            else:
                zone_sto=0
            if adx[i]>25 and d[i-1]<=20 and d[i]>=20:
                adx_sto_kp_signal.append(1)
            elif d[i]<d[i-1] and zone_sto==1 and adx[i]>=40:
                adx_sto_kp_signal.append(-1)
            else:
                adx_sto_kp_signal.append(0)
    
    def stoch_signal_2_func():
        global stoch_signal_2
        stoch_signal_2=[] 
        for i in range(len(dfa1)):
            if k[i] > d[i] and k[i-1] < d[i-1] and k[i]>=80:
                stoch_signal_2.append(-1)
            elif k[i] < d[i] and k[i-1] > d[i-1] and k[i]<=20:
                stoch_signal_2.append(1)
            else:
                stoch_signal_2.append(0)
    
    
    high_low = dfa1_all['High'] - dfa1_all['Low']
    high_close = np.abs(dfa1_all['High'] - dfa1_all['Close'].shift())
    low_close = np.abs(dfa1_all['Low'] - dfa1_all['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).sum()/14
    atr=atr.loc[Sdate:Edate]

    def roc_signal_func():
        global roc_signal
        roc = ta.ROC(dfa1_all["Close"], timeperiod=10)
        roc=roc.loc[Sdate:Edate]
        data=roc
        roc_signal = []
        signal = 0
        for i in range(len(data)):
            if data[i] > 0:
                if signal != 1:
                    signal = 1
                    roc_signal.append(signal)
                else:
                    roc_signal.append(0)
            elif data[i] < 0:
                if signal != -1:
                    signal = -1
                    roc_signal.append(signal)
                else:
                    roc_signal.append(0)
            else:
                roc_signal.append(0)
    global RSI
    RSI = ta.RSI(dfa1_all['Close'], timeperiod=14)
    RSI=RSI.loc[Sdate:Edate]
    RSI=RSI.array

    def rsi_signal_func():
        global rsi_signal
        rsi_signal = []
        signal = 0
        for i in range(len(prices)):
            if RSI[i] <= 20:
                signal=1
            elif RSI[i] >= 80:
                signal=-1
            else:
                signal=0
            rsi_signal.append(signal)

    def rsi_adx_signal_func():
        global rsi_adx_signal
        dfa1['RSI']=RSI
        rsi_adx_signal = []
        signal = 0
        for i in range(len(RSI)):
            if RSI[i-1] <= 30 and RSI[i] >=30 and adx[i]>=25:
                rsi_adx_signal.append(1)
            elif RSI[i] >= 70 and adx[i]<=adx[i-1] and adx[i]>=40:
                rsi_adx_signal.append(-1)
            else:
                rsi_adx_signal.append(0)

    def cci_signal_func():
        global cci_signal
        CCI=ta.CCI(dfa1_all["High"],dfa1_all["Low"],dfa1_all["Close"],timeperiod=14)
        CCI=CCI.loc[Sdate:Edate]
        cci_signal = []
        data=CCI
        signal = 0
        countb,counts =0,0
        for i in range(len(data)):
            if data[i] <= -90:
                counts = 0
                countb = countb+1
                if countb == 6 and signal != 1 :
                    signal = 1
                    cci_signal.append(signal)
                else:
                    cci_signal.append(0)
            elif data[i] >90:
                countb = 0
                counts = counts+1
                if counts == 6 and signal != -1:
                    signal = -1
                    cci_signal.append(signal)
                else:
                    cci_signal.append(0)
            else:
                cci_signal.append(0)
                countb,counts =0,0

    def atr_signal_func():
        global atr_signal
        atr_signal=[0]*len(dfa1)

    def wpr_signal_func():
        global wpr_signal
        wpr=ta.WILLR(dfa1_all["High"],dfa1_all["Low"],dfa1_all["Close"],timeperiod=14)
        wpr=wpr.loc[Sdate:Edate]
        wpr_signal = []
        data=wpr
        signal = 0
        countb,counts =0,0
        for i in range(len(data)):
            if data[i] <= -70:
                counts = 0
                if data[i]<= -80 :
                    countb = countb+1
                if countb == 5 and signal != 1 :
                    signal = 1
                    wpr_signal.append(signal)
                else:
                    wpr_signal.append(0)
            elif data[i] > -30:
                countb = 0
                if data[i]>= -20:
                    counts = counts+1
                if counts == 5 and signal != -1:
                    signal = -1
                    wpr_signal.append(signal)
                else:
                    wpr_signal.append(0)
            else:
                wpr_signal.append(0)
                countb,counts =0,0

    def bb_signal_func():
        global bb_signal
        upperband, middleband, lowerband = ta.BBANDS(dfa1_all['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        upperband, middleband, lowerband=upperband, middleband, lowerband.loc[Sdate:Edate]
        upperband, middleband, lowerband=upperband, middleband, lowerband.array
        dfa1['upperband']=upperband
        dfa1['middleband']=middleband
        dfa1['lowerband']=lowerband
        buy_price = []
        sell_price = []
        bb_signal = []
        signal = 0
        data=dfa1['Open']
        lower_bb=dfa1['lowerband']
        upper_bb=dfa1['upperband']
        a,b,c=dfa1['Close'], dfa1['lowerband'], dfa1['upperband']
        for i in range(len(data)):
            if data[i-1] > lower_bb[i-1] and data[i] < lower_bb[i]:
                if signal != 1:
                    signal = 1
                    bb_signal.append(signal)
                else:
                    bb_signal.append(0)
            elif data[i-1] < upper_bb[i-1] and data[i] > upper_bb[i]:
                if signal != -1:
                    signal = -1
                    bb_signal.append(signal)
                else:   
                    bb_signal.append(0)
            else:
                bb_signal.append(0)

    def cdc_signal_func():
        global cdc_signal
        global color
        ohlc4=(dfa1_all['Close']+dfa1_all['Low']+dfa1_all['High']+dfa1_all['Open'])/4
        EMA2=ta.EMA(ohlc4, timeperiod=2)
        EMA26=ta.EMA(EMA2, timeperiod=26)
        EMA12=ta.EMA(EMA2, timeperiod=12)
        EMA26=EMA26.loc[Sdate:Edate]
        EMA26=EMA26.array
        dfa1['EMA26']=EMA26

        EMA12=EMA12.loc[Sdate:Edate]
        EMA12=EMA12.array
        dfa1['EMA12']=EMA12

        EMA2=EMA2.loc[Sdate:Edate]
        EMA2=EMA2.array
        dfa1['EMA2']=EMA2

        ohlc4=ohlc4.loc[Sdate:Edate]
        ohlc4=ohlc4.array
        dfa1['ohlc4']=ohlc4
        fastslow = []
        for i in range(len(dfa1)):
            if dfa1['EMA12'][i] > dfa1['EMA26'][i]:
                fastslow.append(1)
            elif dfa1['EMA26'][i] > dfa1['EMA12'][i]:
                fastslow.append(-1)
            else:
                fastslow.append(0)

        color = []
        for i in range(len(dfa1)):
            if fastslow[i] == 1 and dfa1['EMA2'][i] > dfa1['EMA12'][i]:
                color.append(1) 
            elif fastslow[i] == -1 and dfa1['EMA2'][i] > dfa1['EMA12'][i]:
                color.append(0.5)
            elif fastslow[i] == -1 and dfa1['EMA2'][i] < dfa1['EMA12'][i]:
                color.append(-1) 
            elif fastslow[i] == 1 and dfa1['EMA2'][i] < dfa1['EMA12'][i]:
                color.append(-0.5) 
            else:
                color.append(0)
        signal=0
        global cdc_signal
        cdc_signal=[]
        for i in range (len(color)):
            if color[i] == 1:
                if signal != 1:
                    signal = 1
                    cdc_signal.append(signal)
                else:
                    cdc_signal.append(0)
            elif color[i] == -1:
                if signal != -1:
                    signal = -1
                    cdc_signal.append(signal)
                else:
                    cdc_signal.append(0)
            elif color[i] == 0.5:
                if signal != 0.5:
                    signal = 0.5
                    cdc_signal.append(signal)
                else:
                    cdc_signal.append(0)
            elif color[i] == -0.5:
                if signal != -0.5:
                    signal = -0.5
                    cdc_signal.append(signal)
                else:
                    cdc_signal.append(0)
            else:
                signal=0
                cdc_signal.append(signal)

    def sar_signal_func():
        global sar_signal
        SAR=ta.SAR(dfa1_all['High'], dfa1_all['Low'], acceleration=0.02, maximum=0.2)
        SAR=SAR.loc[Sdate:Edate]
        SAR=SAR.array
        dfa1['SAR']=SAR
        signal= 0
        countb,counts =0,0
        sar_signal = []
        for i in range(len(dfa1)):
            if dfa1['Close'][i] > dfa1['SAR'][i]:
                counts = 0
                countb =countb +1
                if countb == 2:
                    signal = 1
                    sar_signal.append(signal)
                else:
                    sar_signal.append(0)
            elif dfa1['Close'][i] < dfa1['SAR'][i]:
                countb = 0
                counts =counts +1
                if counts == 2:
                    signal = -1
                    sar_signal.append(signal)
                else:
                    sar_signal.append(0)
            else:
                sar_signal.append(0)

    def aroon_signal_func():
        global aroon_signal
        down,up=ta.AROON(dfa1_all['High'], dfa1_all['Low'], timeperiod=14)
        down=down.loc[Sdate:Edate]
        up=up.loc[Sdate:Edate]
        down=down.array
        up=up.array
        aroon_signal = []
        signal = 0
        for i in range(len(dfa1)):
            if up[i] >= 70 and down[i] <= 30:
                if signal != 1:
                    signal = 1
                    aroon_signal.append(signal)
                else:
                    aroon_signal.append(0)
            elif up[i] <= 30 and down[i] >= 70:
                if signal != -1:
                    signal = -1
                    aroon_signal.append(signal)
                else:
                    aroon_signal.append(0)
            else:
                aroon_signal.append(0)

    def rsiandmacd_signal_func():
        global rsiandmacd_signal
        rsiandmacd_signal=[]
        for i in range(len(dfa1)):
            if i!=0:
                if prices[i] <= 35 and macd_crossover_signal[i]==1 and prices[i]>prices[i-1]:
                    rsiandmacd_signal.append(1)
                elif prices[i] >= 65 and macd_crossover_signal[i]==-1 and prices[i]<=prices[i-1]:
                    rsiandmacd_signal.append(-1)
                else:
                    rsiandmacd_signal.append(0)
            else:
                    rsiandmacd_signal.append(0)
    def renko_signal_func():
        global renko_signal
        global renko_start
        renko_signal=[]
        renko_brick=[]
        renko_start=dfa1['Close'][0]
        for i in range(len(dfa1)):
            if i!=0:
                if dfa1['Close'][i]>=1.01*renko_start:
                    renko_brick.append(1)
                    renko_start=dfa1['Close'][i]
                elif dfa1['Close'][i]<=0.99*renko_start:
                    renko_brick.append(-1)
                    renko_start=dfa1['Close'][i]
                else:
                    renko_brick.append(0)
                if renko_brick[i]==1 and renko_brick[i-1]==1:
                    renko_signal.append(1)
                elif renko_brick[i]==-1 and renko_brick[i-1]==-1:
                    renko_signal.append(-1)
                else:
                    renko_signal.append(0)
            else:
                renko_brick.append(0)
                renko_signal.append(0)
        
    def renko_macd_crossover_signal_func():
        global renko_macd_crossover_signal
        macd, macdsignal, macdhist = ta.MACD(dfa1_all['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        MACD=macd.loc[Sdate:Edate]
        dfa1['MACD']=MACD
        data_macd=dfa1
        data=dfa1
        prices=dfa1['Open']
        renko_macd_crossover_signal=[]
        for i in range(len(dfa1)):
            if i>=4:
                if renko_signal[i]==1 and data_macd['MACD'][i] > data_macd['macdsignal'][i] and data_macd['MACD'][i-1] > data_macd['macdsignal'][i-1] and data_macd['MACD'][i-2] > data_macd['macdsignal'][i-2] and data_macd['MACD'][i-3] > data_macd['macdsignal'][i-3] and data_macd['MACD'][i-4] > data_macd['macdsignal'][i-4]:
                    renko_macd_crossover_signal.append(1)
                elif renko_signal[i]==-1 and data_macd['MACD'][i] < data_macd['macdsignal'][i] and data_macd['MACD'][i-1] < data_macd['macdsignal'][i-1] and data_macd['MACD'][i-2] < data_macd['macdsignal'][i-2]and data_macd['MACD'][i-3] < data_macd['macdsignal'][i-3]and data_macd['MACD'][i-4] < data_macd['macdsignal'][i-4]:
                    renko_macd_crossover_signal.append(-1)
                else:
                    renko_macd_crossover_signal.append(0)
            else:
                renko_macd_crossover_signal.append(0)

    def renko_macdzero_signal_func(): 
        global renko_macdzero_signal
        renko_macdzero_signal=[]
        macd, macdsignal, macdhist = ta.MACD(dfa1_all['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        MACD=macd.loc[Sdate:Edate]
        dfa1['MACD']=MACD
        data_macd=dfa1
        for i in range(len(dfa1)):
            if i>=4:
                if renko_signal[i]==1 and data_macd['MACD'][i] > 0:
                    renko_macdzero_signal.append(1)
                elif renko_signal[i]==-1 and data_macd['MACD'][i] < 0:
                    renko_macdzero_signal.append(-1)
                else:
                    renko_macdzero_signal.append(0)
            else:
                renko_macdzero_signal.append(0)
     
    def fake_obv_renko_signal_func(): 
        global fake_obv_renko_signal,fake_price
        fake_obv_renko_signal=[]
        fake_price=ta.LINEARREG_SLOPE(dfa1_all['Close'],timeperiod=14)
        fake_price=fake_price.loc[Sdate:Edate]
        for i in range(len(dfa1)):
            if fake_price[i]<=0.3 and renko_signal[i]==1:
                fake_obv_renko_signal.append(1)
            elif fake_price[i]>0.3 and renko_signal[i]==-1:
                fake_obv_renko_signal.append(-1)
            else:
                fake_obv_renko_signal.append(0)
    
    def fake_obv_macd_crossover_signal_func():
        global fake_obv_macd_crossover_signal
        fake_obv_macd_crossover_signal=[]
        macd, macdsignal, macdhist = ta.MACD(dfa1_all['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        MACD=macd.loc[Sdate:Edate]
        dfa1['MACD']=MACD
        data_macd=dfa1
        fake_price=ta.LINEARREG_SLOPE(dfa1_all['Close'],timeperiod=14)
        fake_price=fake_price.loc[Sdate:Edate]
        for i in range(len(dfa1)):
            if fake_price[i]<=0.3 and data_macd['MACD'][i] > data_macd['macdsignal'][i] and data_macd['MACD'][i-1] > data_macd['macdsignal'][i-1] and data_macd['MACD'][i-2] > data_macd['macdsignal'][i-2] and data_macd['MACD'][i-3] > data_macd['macdsignal'][i-3] and data_macd['MACD'][i-4] > data_macd['macdsignal'][i-4]:
                fake_obv_macd_crossover_signal.append(1)
            elif fake_price[i]>0.3 and data_macd['MACD'][i] < data_macd['macdsignal'][i] and data_macd['MACD'][i-1] < data_macd['macdsignal'][i-1] and data_macd['MACD'][i-2] < data_macd['macdsignal'][i-2]and data_macd['MACD'][i-3] < data_macd['macdsignal'][i-3]and data_macd['MACD'][i-4] < data_macd['macdsignal'][i-4]:
                fake_obv_macd_crossover_signal.append(-1)
            else:
                fake_obv_macd_crossover_signal.append(0)

    def real_obv_macd_crossover_signal_func():
        global real_obv_macd_crossover_signal,real_price
        macd, macdsignal, macdhist = ta.MACD(dfa1_all['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        MACD=macd.loc[Sdate:Edate]
        dfa1['MACD']=MACD
        data_macd=dfa1
        real_price=ta.LINEARREG_SLOPE(obv,timeperiod=14)
        real_price=real_price.loc[Sdate:Edate]
        real_obv_macd_crossover_signal=[]  
        for i in range(len(dfa1)):
            if real_price[i]>=0.3 and data_macd['MACD'][i] > data_macd['macdsignal'][i] and data_macd['MACD'][i-1] > data_macd['macdsignal'][i-1] and data_macd['MACD'][i-2] > data_macd['macdsignal'][i-2] and data_macd['MACD'][i-3] > data_macd['macdsignal'][i-3] and data_macd['MACD'][i-4] > data_macd['macdsignal'][i-4]:
                real_obv_macd_crossover_signal.append(1)
            elif real_price[i]<0.3 and data_macd['MACD'][i] < data_macd['macdsignal'][i] and data_macd['MACD'][i-1] < data_macd['macdsignal'][i-1] and data_macd['MACD'][i-2] < data_macd['macdsignal'][i-2]and data_macd['MACD'][i-3] < data_macd['macdsignal'][i-3]and data_macd['MACD'][i-4] < data_macd['macdsignal'][i-4]:
                real_obv_macd_crossover_signal.append(-1)
            else:
                real_obv_macd_crossover_signal.append(0)
    def stoch_adx_signal_func():  
        global stoch_adx_signal     
        stoch_adx_signal=[]
        for i in range(len(dfa1)):
            if ADX[i]>35:
                if stoch_signal[i]==1:
                    stoch_adx_signal.append(1)
                elif stoch_signal[i]==-1:
                    stoch_adx_signal.append(-1)
                else:
                    stoch_adx_signal.append(0)
            else:
                stoch_adx_signal.append(0)
    def bb_rsi_signal_func():
        global bb_rsi_signal
        bb_rsi_signal=[]
        for i in range(len(data)):
            if (bb_signal[i]==1 and prices[i]>=50):
                bb_rsi_signal.append(1)
            elif (bb_signal[i]==-1 and prices[i]<=40):
                bb_rsi_signal.append(-1)
            else:
                bb_rsi_signal.append(0)

    def psar_wpr_signal_func():
        global psar_wpr_signal
        psar_wpr_signal=[]
        for i in range(len(data)):
            if (psar_changetrend_signal[i]==1 and wpr_signal[i]!=1):
                psar_wpr_signal.append(1)
            elif (psar_changetrend_signal[i]==-1 and wpr_signal[i]!=-1):
                psar_wpr_signal.append(-1)
            else:
                psar_wpr_signal.append(0)

    def cdc_atr_signal_func():
        global cdc_atr_signal 
        cdc_atr_signal=[]
        for i in range(len(data)):
            if (cdc_signal[i]==1 and atr_signal[i]==1):
                cdc_atr_signal.append(1)
            elif (cdc_signal[i]==-1 and atr_signal[i]==1):
                cdc_atr_signal.append(-1)
            else:
                cdc_atr_signal.append(0)
        
    def macd_sto_signal_func():
        global macd_sto_signal
        macd_sto_signal=[]
        for i in range(len(data)):
            if (stoch_signal[i]==1 and macd_crossover_signal[i]==1):
                macd_sto_signal.append(1)
            elif (stoch_signal[i]==-1 and macd_crossover_signal[i]==-1):
                macd_sto_signal.append(-1)
            else:
                macd_sto_signal.append(0)
            
    def sto_ma50_200_signal_func():
        global sto_ma50_200_signal 
        sto_ma50_200_signal=[]
        for i in range(len(data)):
            if (stoch_signal[i]==1 and ma50_200_signal[i]==1):
                sto_ma50_200_signal.append(1)
            elif (stoch_signal[i]==-1 and ma50_200_signal[i]==-1):
                sto_ma50_200_signal.append(-1)
            else:
                sto_ma50_200_signal.append(0)

    def atr_adx_signal_func():  
        global atr_adx_signal   
        atr_adx_signal=[]
        for i in range(len(data)):
            if (atr_signal[i]==1 and adx_signal[i]==1):
                atr_adx_signal.append(1)
            elif (atr_signal[i]==-1 and adx_signal[i]==-1):
                atr_adx_signal.append(-1)
            else:
                atr_adx_signal.append(0)

    def rsi_divergent_signal_func():
        global rsi_divergent_signal
        data=rsiuse(dfa1)
        rsi_divergent_signal=rsidiver()
    
    def psar_adx_signal_func():
        global psar_adx_signal
        psar_adx_signal=[]
        for i in range(len(data)):
            if (psar_changetrend_signal[i]==1 and ADX[i]>40 and ADX[i]<ADX[i-1]):
                psar_adx_signal.append(1)
            elif (psar_changetrend_signal[i]==-1 and ADX[i]>25 and ADX[i]<ADX[i-1]):
                psar_adx_signal.append(-1)
            else:
                psar_adx_signal.append(0)
            
    def cci_renko_signal_func():
        global cci_renko_signal
        cci_renko_signal=[]
        for i in range(len(data)):
            if (renko_signal[i]==1 and RSI[i] <= 20):
                cci_renko_signal.append(1)
            elif (renko_signal[i]==-1 and RSI[i] >= 80):
                cci_renko_signal.append(-1)
            else:
                cci_renko_signal.append(0)
    
    def macd_crossover_wpr_signal_func():
        global macd_crossover_wpr_signal
        macd_crossover_wpr_signal=[]
        wpr=ta.WILLR(dfa1_all["High"],dfa1_all["Low"],dfa1_all["Close"],timeperiod=14)
        wpr=wpr.loc[Sdate:Edate]
        for i in range(len(data)):
            if (macd_crossover_signal[i]==1 and wpr[i] <= -80):
                macd_crossover_wpr_signal.append(1)
            elif (macd_crossover_signal[i]==-1 and wpr[i] <= -20):
                macd_crossover_wpr_signal.append(-1)
            else:
                macd_crossover_wpr_signal.append(0)

    def aroon_kp_signal_func():
        global aroon_kp_signal
        aroon_kp_signal=[]
        down,up=ta.AROON(dfa1_all['High'], dfa1_all['Low'], timeperiod=14)
        down=down.loc[Sdate:Edate]
        up=up.loc[Sdate:Edate]
        down=down.array
        for i in range(len(data)):
            if (up[i-1]<down[i-1] and up[i]>down[i]):
                aroon_kp_signal.append(1)
            elif (up[i-1]>down[i-1] and up[i]<down[i]):
                aroon_kp_signal.append(-1)
            else:
                aroon_kp_signal.append(0)

    def renko_adx_signal_func():
        global renko_adx_signal
        renko_adx_signal=[]
        for i in range(len(data)):
            if (renko_signal[i]==1 and ADX[i]>35):
                renko_adx_signal.append(1)
            elif (renko_signal[i]==-1 and ADX[i]<25):
                renko_adx_signal.append(-1)
            else:
                renko_adx_signal.append(0)
            
    def volume_prof_signal_func():
        global volume_prof_signal,volume
        signal=0
        volume_prof_signal=[]
        data=dfa1
        volume = data['Volume']
        close = data['Close']
        kde_factor = 0.05
        num_samples = 500
        kde = stats.gaussian_kde(close,weights=volume,bw_method=kde_factor)
        xr = np.linspace(close.min(),close.max(),num_samples)
        kdy = kde(xr)
        ticks_per_sample = (xr.max() - xr.min()) / num_samples
        min_prom = 1
        min_prom = kdy.max() * 0.3
        peaks, peak_props = sg.find_peaks(kdy, prominence=min_prom)
        pkx = xr[peaks]
        pky = kdy[peaks]
        pkx=sorted(pkx,reverse=True)
        for i in range(len(data)):
            if len(pkx)==1:
                if close[i] > pkx[0]:
                    if signal != 1:
                        signal = 1
                        volume_prof_signal.append(signal)
                    else:
                        volume_prof_signal.append(0)
                elif close[i] < pkx[0]:
                    if signal != -1:
                        signal = -1
                        volume_prof_signal.append(signal)
                    else:
                        volume_prof_signal.append(0)
                else:
                    volume_prof_signal.append(0)
            else:
                max=pkx[0]
                min=pkx[1]
                if (close[i]<min):
                    volume_prof_signal.append(1)
                elif (close[i]>max):
                    volume_prof_signal.append(-1)
                else:
                    volume_prof_signal.append(0)
                # if close[i] > pkx[1]: # 3
                #     if signal != 1:
                #         signal = 1
                #         volume_prof_signal.append(signal)
                #     else:
                #         volume_prof_signal.append(0)
                # elif close[i] < pkx[1]:
                #     if signal != -1:
                #         signal = -1
                #         volume_prof_signal.append(signal)
                #     else:
                #         volume_prof_signal.append(0)
                # else:
                #     volume_prof_signal.append(0)
                # if close[i] < pkx[0]: #max 5
                #     if signal != 1:
                #         signal = 1
                #         volume_prof_signal.append(signal)
                #     else:
                #         volume_prof_signal.append(0)
                # elif close[i] > pkx[0]:
                #     if signal != -1:
                #         signal = -1
                #         volume_prof_signal.append(signal)
                #     else:
                #         volume_prof_signal.append(0)
                # else:
                #     volume_prof_signal.append(0)
    def vwap_signal_func():
        global vwap_signal 
        vwap_signal=[]
        df=dfa1
        v = df['Volume'].values
        tp = (df['Low'] + df['Close'] + df['High']).div(3).values
        df=df.assign(vwap=(tp * v).cumsum() / v.cumsum())
        vwap=df['vwap']
        signal=0
        for i in range(len(data)):
            if vwap[i]>dfa1['Close'][i]:
                if signal != 1:
                    signal = 1
                    vwap_signal.append(signal)
                else:
                    vwap_signal.append(0)
            elif vwap[i]<dfa1['Close'][i] :
                if signal != -1:
                    signal = -1
                    vwap_signal.append(signal)
                else:
                    vwap_signal.append(0)
            else:
                vwap_signal.append(0)
                

    adx_signal_func()
    macd_crossover_signal_func()
    ma30_200_signal_func()
    ma50_200_signal_func()
    ma30_50_signal_func()
    cdc_signal_func()
    sar_signal_func()
    roc_signal_func()
    obv_signal_func()
    rsi_signal_func()
    stoch_signal_func()
    cci_signal_func()
    sma100_signal_func()
    atr_signal_func()
    renko_signal_func()
    ma50_100_signal_func()
    sma89_signal_func()
    ma30_100_signal_func()
    stoch_adx_signal_func()
    ma5_10_signal_func()
    ma25_89_signal_func()
    ma21_89_signal_func()
    macd_overzero_signal_func()
    rsi_divergent_signal_func()
    rsi_adx_signal_func()
    psar_adx_signal_func()
    cci_renko_signal_func()
    renko_adx_signal_func()
    volume_prof_signal_func()
    bb_signal_func()
    bb_rsi_signal_func()
    real_obv_macd_crossover_signal_func()
    renko_macd_crossover_signal_func()
    renko_macdzero_signal_func()
    aroon_kp_signal_func()
    #vwap_signal_func()
    signalss={'adx_signal':adx_signal,'macd_crossover_signal':macd_crossover_signal,
            'ma30_200_signal':ma30_200_signal,'ma50_200_signal':ma50_200_signal,
            'ma30_50_signal':ma30_50_signal,'cdc_signal':cdc_signal,'sar_signal':sar_signal,
            'roc_signal':roc_signal,'obv_signal':obv_signal,'rsi_signal':rsi_signal,
            'stoch_signal':stoch_signal,'atr_signal':atr_signal,'bb_signal':bb_signal,
            'psar_changetrend_signal':psar_changetrend_signal,
            'renko_signal':renko_signal,'renko_macd_crossover_signal':renko_macd_crossover_signal,'ma50_100_signal':ma50_100_signal,
            'ma30_100_signal':ma30_100_signal,'stoch_adx_signal':stoch_adx_signal,
            'bb_rsi_signal':bb_rsi_signal,'ma5_10_signal':ma5_10_signal,
            'ma25_89_signal':ma25_89_signal,'ma21_89_signal':ma21_89_signal
            ,'macd_overzero_signal':macd_overzero_signal,'rsi_divergent_signal':rsi_divergent_signal,
            'sma89_signal':sma89_signal,'sma100_signal':sma100_signal,
            'real_obv_macd_crossover_signal':real_obv_macd_crossover_signal,
            'renko_macdzero_signal':renko_macdzero_signal,'psar_adx_signal':psar_adx_signal,
            'aroon_kp_signal':aroon_kp_signal,'renko_adx_signal':renko_adx_signal,
            'volume_prof_signal':volume_prof_signal,
            'close_price':dfa1['Close'],'rsi_adx_signal':rsi_adx_signal,
           'adj_close_price':dfa1['Adj Close']}
    
    # globals()['df_of_'+Real_name+'_signal'+'_train']=pd.DataFrame(signalss)
    global c 
    c = pd.DataFrame(signalss) 
    return c

def get_ma89_value():
    return MA89
def get_ma100_value():
    return MA100
def get_adx_value():
    return ADX
def get_atr_value():
    return atr
def get_rsi_value():
    return RSI
def get_macd_value():
    return MACD
def get_cdc_value():
    return color
def get_stoch_value():
    return k
def get_psar_value():
    return psarc
def get_volume_value():
    return volume