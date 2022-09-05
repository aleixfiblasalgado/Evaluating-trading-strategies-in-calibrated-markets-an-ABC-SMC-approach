#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 02:28:22 2022

@author: aleixfiblasalgado
"""

## LOBSTER utils

import pandas as pd
import itertools
import numpy as np


def get_trades(stream_df, start_time = '2021-11-22T09:30:00', 
               end_time = '2021-11-22T10:30:00'):
    
    
    mask = (stream_df['QuoteTime'] >= start_time) & (stream_df['QuoteTime'] <= end_time)
    stream_df = stream_df.loc[mask]
    # df = stream_11_22[(stream_11_22['Event Type'] == 4) | (stream_11_22['Event Type'] == 5)]
    df = stream_df[(stream_df['Event Type'] == 4)]
    df = df.set_index('QuoteTime')
    
    # New code for minutely resampling and renaming columns.
    df = df[["Price","Size"]].resample("1T")
    df_open = df["Price"].first().ffill()
    df_close = df["Price"].last().ffill()
    df_high = df["Price"].max().ffill()
    df_low = df["Price"].min().ffill()
    df_vol = df["Price"].sum()
    ohlcv = pd.DataFrame({
        "open": df_open,
        "high": df_high,
        "low": df_low,
        "close": df_close,
        "volume": df_vol
    })
  

    return ohlcv


def stream_cleaning(stream_df, start_time = '2021-11-22T09:30:00', 
               end_time = '2021-11-22T10:30:00'):
    
    
    mask = (stream_df['QuoteTime'] >= start_time) & (stream_df['QuoteTime'] <= end_time)
    stream_df = stream_df.loc[mask]
    # df = stream_11_22[(stream_11_22['Event Type'] == 4) | (stream_11_22['Event Type'] == 5)]
    df = stream_df[(stream_df['Event Type'] == 1) | (stream_df['Event Type'] == 2) |
                   (stream_df['Event Type'] == 3) | (stream_df['Event Type'] == 4) |
                   (stream_df['Event Type'] == 5)]
    
    cols = df.columns.tolist()
    cols = ['QuoteTime', 'Order ID', 'Price', 'Size', 'Direction', 'Event Type']
    df = df[cols]
    df.columns = ['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'TYPE']
  

    return df


def orderbook_cleaning(orderbook_df, num_levels):
    
    colnum = num_levels*4
    processed_ob = orderbook_df.iloc[:,0:colnum]
  
    return processed_ob 


def get_orderbook_stylised_facts(stream_processed, orderbook_df, num_levels=5, 
                                 order_type="executed", freq="20S"):
    
    columns = list(itertools.chain(
    *[[f'ask_price_{level}', f'ask_size_{level}', f'bid_price_{level}', f'bid_size_{level}'] for level in
      range(1, num_levels + 1)]))
    
    merged = pd.merge(stream_processed, orderbook_df, left_index=True, right_index=True, how='left')
    merged = merged.set_index("TIMESTAMP")
    merge_cols = ['ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'TYPE'] + columns
    merged = merged[merge_cols]
    merged['PRICE'] = merged['PRICE'] / 10000
    
    ignore_cancellations=True

    # merged = merged.dropna()
    merged = merged.ffill()
    
    # Ignore cancellations
    if ignore_cancellations:
        merged = merged[merged.TYPE != 2]
        merged = merged[merged.TYPE != 3]
        
    merged['MID_PRICE'] = (merged['ask_price_1'] + merged['bid_price_1']) / (2 * 10000)
    merged['SPREAD'] = (merged['ask_price_1'] - merged['bid_price_1']) / 10000
    merged['TOTAL_VOLUME'] = merged['ask_size_1'] + merged['bid_size_1']
    merged['VOLUME_L1_IMBALANCE'] = merged['ask_size_1'] - merged['bid_size_1']
    merged['ORDER_VOLUME_IMBALANCE'] = merged['ask_size_1'] / (merged['bid_size_1'] + merged['ask_size_1'])
    
    transacted_orders = merged.loc[(merged.TYPE == 4) | (merged.TYPE == 5)]
    limit_orders = merged.loc[merged.TYPE == 1]
    
    if order_type=="executed":
        # transacted_orders = transacted_orders.loc[:,['MID_PRICE','SPREAD','TOTAL_VOLUME','VOLUME_L1_IMBALANCE', 'ORDER_VOLUME_IMBALANCE']].resample(freq)
        transacted_orders = transacted_orders.loc[:,['ask_price_1','bid_price_1',
                                                     'ask_size_1', 'bid_size_1', 'MID_PRICE', 'SPREAD']].resample(freq)
        
        # mid_price_w = ((transacted_orders['ask_price_1'].last() + transacted_orders['bid_price_1'].last()) / (2 * 100)).ffill()
        # spread_w = ((transacted_orders['ask_price_1'].last() - transacted_orders['bid_price_1'].last()) / 100).ffill()
        # mid_price_w = transacted_orders['MID_PRICE'].mean().ffill()
        # spread_w = transacted_orders['SPREAD'].mean().ffill()
        volume_w = (transacted_orders['ask_size_1'].sum() + transacted_orders['bid_size_1'].sum()).ffill()
        # l1_imbalance_w = (transacted_orders['ask_size_1'].sum() - transacted_orders['bid_size_1'].sum()).ffill()        
        ov_imbalance_w = (transacted_orders['ask_size_1'].sum()/(transacted_orders['ask_size_1'].sum() + transacted_orders['bid_size_1'].sum())).ffill()  
    
    else:
        transacted_orders = transacted_orders.loc[:,['ask_price_1','bid_price_1',
                                                     'ask_size_1', 'bid_size_1', 'MID_PRICE', 'SPREAD']].resample(freq)
        limit_orders = transacted_orders.loc[:,['ask_price_1','bid_price_1',
                                                     'ask_size_1', 'bid_size_1', 'MID_PRICE', 'SPREAD']].resample(freq)
        # mid_price_w = ((transacted_orders['ask_price_1'].last() + transacted_orders['bid_price_1'].last()) / (2 * 100)).ffill()
        # spread_w = ((transacted_orders['ask_price_1'].last() - transacted_orders['bid_price_1'].last()) / 100).ffill()
        # mid_price_w = transacted_orders['MID_PRICE'].mean().ffill()
        # spread_w = transacted_orders['SPREAD'].mean().ffill()
        volume_w = (limit_orders['ask_size_1'].sum() + limit_orders['bid_size_1'].sum()).ffill()
        # l1_imbalance_w = (limit_orders['ask_size_1'].sum() - limit_orders['bid_size_1'].sum()).ffill()        
        ov_imbalance_w = (limit_orders['ask_size_1'].sum()/(limit_orders['ask_size_1'].sum() + transacted_orders['bid_size_1'].sum())).ffill()
        
    lob_stylized_facts = pd.DataFrame({
    # "mid_price": mid_price_w,
    # "spread": spread_w,
    # "volume_t": volume_w/np.linalg.norm(volume_w),
    "volume_t": volume_w,
    # "volume_s": l1_imbalance_w,
    "imbalance": ov_imbalance_w
    })
    
    return(lob_stylized_facts)


def get_interarrival_times(stream_df):
    stream_df = stream_df.set_index('TIMESTAMP')
    arrival_times = stream_df.index.to_series()
    interarrival_times = arrival_times.diff()
    interarrival_times = interarrival_times.iloc[1:].apply(pd.Timedelta.total_seconds)
    interarrival_times = interarrival_times.rename("Interarrival time /s")
    return(interarrival_times)
