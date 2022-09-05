"""
## Simulator
@author: aleix
"""
import logging
import tempfile


import pyabc
from pyabc.distance import *
from pyabc.predictor import *
from pyabc.sumstat import *
from pyabc.util import EventIxs, ParTrafo, dict2arrlabels
from pyabc.sampler import SingleCoreSampler
pyabc.settings.set_figure_params('pyabc') 
# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import datetime as dt
import functools
import warnings
import pickle
import itertools
import scipy as sp
from dateutil.parser import parse
from IPython.display import SVG, display


from market_simulations import rmsc03


if __name__ == "__main__":
    
    p_true = {"p0": -0.7, "p1": 0.1}
    cov = np.array([[1, 0.5], [0.5, 1]])
    n = 100


    def model(p):
        mean = np.array([p["p0"], p["p1"]])
        # shape (n, 2)
        y = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
        return {"y": y}


    data = model(p_true)
    
    prior = pyabc.Distribution(
    **{par: pyabc.RV("norm", p_true[par], 0.25) for par in p_true}
    )
    prior_bounds = {par: (p_true[par] - 0.7, p_true[par] + 0.7) for par in p_true}
    
    
    class IdSumstat(pyabc.Sumstat):
        """Identity summary statistic."""

        def __call__(self, data: dict) -> np.ndarray:
            # shape (n, dim)
            return data["y"]
        
    # population size too small for practice
    pop_size = 200  
    max_eval = int(2e4)