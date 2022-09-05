#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 01:39:45 2022

@author: aleixfiblasalgado
"""

import logging

import pyabc
pyabc.settings.set_figure_params('pyabc')
# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

## Libraries required
from random import seed
from util.utilities import annotate_axes

import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid") 

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=get_cmap("tab20").colors) 

if __name__ == "__main__":
    
    returns = pd.read_csv("returns.csv")
    returns1 = pd.read_csv("returns1.csv")
    returns = returns.iloc[:,1]
    returns1 = returns1.iloc[:,1]
    returns_all = pd.concat([returns, returns1])
    returns_all = returns_all/10000
    
    figpath = "../thesis_template/figures/"
    sns.distplot(returns_all,  axlabel="$")
    plt.savefig(figpath+'Figure_1_E3.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    