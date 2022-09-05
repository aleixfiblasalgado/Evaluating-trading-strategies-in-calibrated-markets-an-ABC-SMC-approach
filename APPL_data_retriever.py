#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 20:11:09 2022

@author: aleixfiblasalgado
"""
import logging

import pyabc
pyabc.settings.set_figure_params('pyabc')
# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

import json
import os
import numpy as np
from market_data import get_market_data


if __name__ == "__main__":
    
    datapath = "../data/_data_dwn_16_372__AAPL_2021-11-22_2021-11-29_20"
    ob_path = os.path.join(datapath, "AAPL_2021-11-22_24900000_57900000_orderbook_20.csv")
    stream_path = os.path.join(datapath, "AAPL_2021-11-22_24900000_57900000_message_20.csv")
    
    data = get_market_data(stream_path, ob_path, start_time = '2021-11-22T10:30:00', 
                        end_time = '2021-11-22T11:30:00')
    
    np.save("../data/APPL_ss_10-11.npy", data)
    
    data["MinutelyReturns"]
    
    
    