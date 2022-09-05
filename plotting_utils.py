#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:00:20 2022

@author: aleixfiblasalgado

"""

import pyabc
import matplotlib.pyplot as plt

def plot_history(history, ground_truth):
    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=0,
            xmax=1000,
            x='num_value_ag',
            ax=ax,
            label=f"PDF t={t}",
            refval=ground_truth,
        )
    ax.legend()

    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=0,
            xmax=1000,
            x='num_momentum_ag',
            ax=ax,
            label=f"PDF t={t}",
            refval=ground_truth,
        )
    ax.legend()

    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=0,
            xmax=8000,
            x='num_noise_ag',
            ax=ax,
            label=f"PDF t={t}",
            refval=ground_truth,
        )
    ax.legend()
    
    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=0,
            xmax=1,
            x='mm_pov',
            ax=ax,
            label=f"PDF t={t}",
            refval=ground_truth,
        )
    ax.legend()