import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np

import talib.abstract as ta
from scipy.stats import linregress, entropy

import matplotlib.pyplot as plt

def generate_features(df):
    df = df.copy()
    
    # Price transformations
    # df['%-log_price'] = np.log(df['Income'])
    df['%-sqrt_price'] = np.sqrt(df['Income'])
    
    # Momentum Indicators (12 features)
    for period in [7, 14]:
        df[f'%-rsi-{period}'] = ta.RSI(df['Income'], timeperiod=period)
        df[f'%-stoch-k-{period}'] = ta.STOCH(df['Income'], df['Income'], df['Income'], 
                                           fastk_period=period)[0]
        df[f'%-stoch-d-{period}'] = ta.STOCH(df['Income'], df['Income'], df['Income'],
                                           fastk_period=period)[1]
    
    df['%-mom-10'] = ta.MOM(df['Income'], timeperiod=10)
    df['%-mom-20'] = ta.MOM(df['Income'], timeperiod=20)
    df['%-cmo-14'] = ta.CMO(df['Income'], timeperiod=14)
    df['%-roc-10'] = ta.ROC(df['Income'], timeperiod=10)
    df['%-roc-20'] = ta.ROC(df['Income'], timeperiod=20)
    df['%-trix-14'] = ta.TRIX(df['Income'], timeperiod=14)
    
    # MACD components
    macd, macdsignal, macdhist = ta.MACD(df['Income'])
    df['%-macd'] = macd
    df['%-macd_signal'] = macdsignal
    df['%-macd_hist'] = macdhist
    
    # Trend Indicators (12 features)
    for period in [7, 14, 30]:
        df[f'%-sma-{period}'] = ta.SMA(df['Income'], timeperiod=period)
        df[f'%-ema-{period}'] = ta.EMA(df['Income'], timeperiod=period)
    
    df['%-dema-20'] = ta.DEMA(df['Income'], timeperiod=20)
    df['%-tema-20'] = ta.TEMA(df['Income'], timeperiod=20)
    df['%-kama-20'] = ta.KAMA(df['Income'], timeperiod=20)
    df['%-wma-20'] = ta.WMA(df['Income'], timeperiod=20)
    df['%-adx-14'] = ta.ADX(df['Income'], df['Income'], df['Income'], timeperiod=14)
    df['%-cci-14'] = ta.CCI(df['Income'], df['Income'], df['Income'], timeperiod=14)
    df['%-vwma-20'] = (df['Income'] * df['Outcome']).rolling(20).sum() / df['Outcome'].rolling(20).sum()
    
    # Volatility Indicators (10 features)
    periods = [7, 14, 30]
    for period in periods:
        df[f'%-atr-{period}'] = ta.ATR(df['Income'], df['Income'], df['Income'], timeperiod=period)
        # df[f'%-volatility-{period}'] = df['Income'].pct_change().rolling(period).std()
    
    upper, _, lower = ta.BBANDS(df['Income'], timeperiod=20)
    df['%-bb_upper-20'] = upper
    df['%-bb_middle-20'] = (upper + lower) / 2
    df['%-bb_lower-20'] = lower
    df['%-bb_width'] = (upper - lower) / df['%-sma-14']
    df['%-bb_pctb'] = (df['Income'] - lower) / (upper - lower)
    
    # Cycle Indicators (5 features)
    df['%-ht_trendline'] = ta.HT_TRENDLINE(df['Income'])
    sine, leadsine = ta.HT_SINE(df['Income'])
    df['%-ht_sine'] = sine
    df['%-ht_leadsine'] = leadsine
    df['%-phase_shift'] = sine - leadsine
    
    # Volume Indicators (5 features)
    df['%-obv'] = ta.OBV(df['Income'], df['Outcome'])  # Fixed to use Outcome as volume
    # df['%-vpt'] = (df['Outcome'] * (df['Income'].pct_change())).cumsum()
    # df['%-volume_change'] = df['Outcome'].pct_change()
    df['%-volume_zscore'] = (df['Outcome'] - df['Outcome'].rolling(20).mean()) / df['Outcome'].rolling(20).std()
    
    # Statistical Features (12 features)
    # df['%-returns-1'] = df['Income'].pct_change()
    # df['%-log_returns'] = np.log(df['Income']).diff()
    for period in [14, 30]:
        df[f'%-rolling_skew-{period}'] = df['Income'].rolling(period).skew()
        df[f'%-rolling_kurt-{period}'] = df['Income'].rolling(period).kurt()
        df[f'%-rolling_q25-{period}'] = df['Income'].rolling(period).quantile(0.25)
        df[f'%-rolling_q75-{period}'] = df['Income'].rolling(period).quantile(0.75)
    
    # df['%-cum_returns'] = (1 + df['Income'].pct_change()).cumprod()
    df['%-expanding_mean'] = df['Income'].expanding().mean()
    df['%-expanding_std'] = df['Income'].expanding().std()
    
    # Time-based Features
    df['%-day_of_week'] = df['Date'].dt.dayofweek
    df['%-month'] = df['Date'].dt.month
    df['%-quarter'] = df['Date'].dt.quarter
    df['%-year'] = df['Date'].dt.year
    
    # Advanced Features
    df['%-chaikin_vol'] = (df['%-bb_upper-20'] - df['%-bb_lower-20']) / df['%-sma-14']
    # df['%-price_intensity'] = df['%-rsi-14'] * df['%-volatility-14']
    df['%-trend_strength'] = df['%-adx-14'] * df['%-cci-14']
    
    # Lagged Features backward
    for lag in [1, 2, 3, 5]:
        df[f'%-lag_{lag}'] = df['Income'].shift(-lag)
    
    # Difference Features
    for diff in [1, 3, 5]:
        df[f'%-diff-{diff}'] = df['Income'].diff(diff)
    
    
    return df