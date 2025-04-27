import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np

import talib.abstract as ta
from scipy.stats import linregress, entropy

import matplotlib.pyplot as plt

def generate_features(df):
    df = df.copy()
    
    for price_col in ['Income', 'Outcome']:
        suffix = 'income' if price_col == 'Income' else 'outcome'
        volume_col = 'Outcome' if price_col == 'Income' else 'Income'
        
        # Price transformations
        df[f'%-log_price_{suffix}'] = np.log1p(df[price_col])
        df[f'%-sqrt_price_{suffix}'] = np.sqrt(df[price_col])
        
        # Momentum Indicators (12 features)
        for period in [2,3,4,5, 7, 14]:
            df[f'%-rsi-{period}_{suffix}'] = ta.RSI(df[price_col], timeperiod=period)
            stoch_k, stoch_d = ta.STOCH(df[price_col], df[price_col], df[price_col], fastk_period=period)
            df[f'%-stoch-k-{period}_{suffix}'] = stoch_k
            df[f'%-stoch-d-{period}_{suffix}'] = stoch_d
        
        df[f'%-mom-10_{suffix}'] = ta.MOM(df[price_col], timeperiod=10)
        df[f'%-mom-20_{suffix}'] = ta.MOM(df[price_col], timeperiod=20)
        df[f'%-cmo-14_{suffix}'] = ta.CMO(df[price_col], timeperiod=14)
        df[f'%-roc-10_{suffix}'] = ta.ROC(df[price_col], timeperiod=10)
        df[f'%-roc-20_{suffix}'] = ta.ROC(df[price_col], timeperiod=20)
        df[f'%-trix-14_{suffix}'] = ta.TRIX(df[price_col], timeperiod=14)
        
        # MACD components
        macd, macdsignal, macdhist = ta.MACD(df[price_col])
        df[f'%-macd_{suffix}'] = macd
        df[f'%-macd_signal_{suffix}'] = macdsignal
        df[f'%-macd_hist_{suffix}'] = macdhist
        
        # Trend Indicators (12 features)
        for period in [2,3,4,5, 7, 14, 30]:
            df[f'%-sma-{period}_{suffix}'] = ta.SMA(df[price_col], timeperiod=period)
            df[f'%-ema-{period}_{suffix}'] = ta.EMA(df[price_col], timeperiod=period)
        
        df[f'%-dema-20_{suffix}'] = ta.DEMA(df[price_col], timeperiod=20)
        df[f'%-tema-20_{suffix}'] = ta.TEMA(df[price_col], timeperiod=20)
        df[f'%-kama-20_{suffix}'] = ta.KAMA(df[price_col], timeperiod=20)
        df[f'%-wma-20_{suffix}'] = ta.WMA(df[price_col], timeperiod=20)
        df[f'%-adx-14_{suffix}'] = ta.ADX(df[price_col], df[price_col], df[price_col], timeperiod=14)
        df[f'%-cci-14_{suffix}'] = ta.CCI(df[price_col], df[price_col], df[price_col], timeperiod=14)
        df[f'%-vwma-20_{suffix}'] = (df[price_col] * df[volume_col]).rolling(20).sum() / df[volume_col].rolling(20).sum()
        
        # Volatility Indicators (10 features)
        periods = [1,2,3,4,5, 7, 14, 30]
        for period in periods:
            df[f'%-atr-{period}_{suffix}'] = ta.ATR(df[price_col], df[price_col], df[price_col], timeperiod=period)
        
        upper, middle, lower = ta.BBANDS(df[price_col], timeperiod=20)
        df[f'%-bb_upper-20_{suffix}'] = upper
        df[f'%-bb_middle-20_{suffix}'] = (upper + lower) / 2
        df[f'%-bb_lower-20_{suffix}'] = lower
        df[f'%-bb_width_{suffix}'] = (upper - lower) / df[f'%-sma-14_{suffix}']
        df[f'%-bb_pctb_{suffix}'] = (df[price_col] - lower) / (upper - lower)
        
        # Cycle Indicators (5 features)
        df[f'%-ht_trendline_{suffix}'] = ta.HT_TRENDLINE(df[price_col])
        sine, leadsine = ta.HT_SINE(df[price_col])
        df[f'%-ht_sine_{suffix}'] = sine
        df[f'%-ht_leadsine_{suffix}'] = leadsine
        df[f'%-phase_shift_{suffix}'] = sine - leadsine
        
        # Volume Indicators (5 features)
        df[f'%-obv_{suffix}'] = ta.OBV(df[price_col], df[volume_col])
        df[f'%-volume_zscore_{suffix}'] = (df[volume_col] - df[volume_col].rolling(20).mean()) / df[volume_col].rolling(20).std()
        
        # Statistical Features (12 features)
        for period in [4, 5, 7, 14, 30]:
            df[f'%-rolling_skew-{period}_{suffix}'] = df[price_col].rolling(period).skew()
            df[f'%-rolling_kurt-{period}_{suffix}'] = df[price_col].rolling(period).kurt()
            df[f'%-rolling_q25-{period}_{suffix}'] = df[price_col].rolling(period).quantile(0.25)
            df[f'%-rolling_q75-{period}_{suffix}'] = df[price_col].rolling(period).quantile(0.75)
        
        df[f'%-expanding_mean_{suffix}'] = df[price_col].expanding().mean()
        df[f'%-expanding_std_{suffix}'] = df[price_col].expanding().std()
        
        # Advanced Features
        df[f'%-chaikin_vol_{suffix}'] = (df[f'%-bb_upper-20_{suffix}'] - df[f'%-bb_lower-20_{suffix}']) / df[f'%-sma-14_{suffix}']
        df[f'%-trend_strength_{suffix}'] = df[f'%-adx-14_{suffix}'] * df[f'%-cci-14_{suffix}']
        
        # Lagged Features backward
        for lag in [1, 2, 3, 5, 7, 14]:
            df[f'%-lag_{lag}_{suffix}'] = df[price_col].shift(-lag)
        
        # Difference Features
        for diff in [1, 2, 3, 4, 5, 7, 14]:
            df[f'%-diff-{diff}_{suffix}'] = df[price_col].diff(diff)

    df[f'%-day_of_week'] = df['Date'].dt.dayofweek
    df[f'%-month'] = df['Date'].dt.month
    df[f'%-quarter'] = df['Date'].dt.quarter
    df[f'%-year'] = df['Date'].dt.year
    
    return df