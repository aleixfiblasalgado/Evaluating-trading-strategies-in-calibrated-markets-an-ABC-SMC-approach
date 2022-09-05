#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 02:31:10 2022

@author: aleixfiblasalgado
"""

"""
## Simulator
@author: aleix
"""
import logging

import pyabc
pyabc.settings.set_figure_params('pyabc')
# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

import numpy as np
import pandas as pd
import os
import tempfile
import datetime as dt
import scipy as sp

from joblib import Parallel, delayed
from market_simulations import exp_agent as market
import multiprocessing

parameters = {
    "num_value_ag":681, 
    "num_momentum_ag":248,
    "num_noise_ag":486,
    "mm_pov":0.18,
    "kappa":3.8e-16,
    "fund_vol":6e-08,
    "megashock_lambda_a":4e-18,
    "megashock_mean":1849,
    "megashock_var":53423,
    "ea_short_window":120,
    "ea_long_window":600
    }

ret = []

def process(seed):
    ea = market(parameters, seed)
    return(ea)

ret = Parallel(n_jobs=8)(delayed(process)(seed) for seed in range(100, 110))
print(ret)

dictionary = {"experimental_agent":ret}
df = pd.DataFrame(dictionary)
df.to_csv('returns.csv') 
