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

from market_simulations import rmsc03_4 as market
from plotting_utils import plot_history

from mlxtend.plotting import ecdf

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=get_cmap("tab20").colors) 

if __name__ == "__main__":

    params_true={"num_value_ag":100 , "num_momentum_ag":25, "num_noise_ag":5000, 
                  "mm_pov":0.025}
    
    data = market(params_true)
    
    priors = pyabc.Distribution(
        num_value_ag=pyabc.RV('randint', 20, 800),
        num_momentum_ag=pyabc.RV('randint', 10, 500),
        num_noise_ag=pyabc.RV('randint', 10, 8000),
        mm_pov=pyabc.RV("uniform", 0.0001, 0.5),
    )
    
    db_path = os.path.join(os.getcwd(), "ABC_output")
    
    abc_ks = pyabc.ABCSMC(market, priors)
    abc_war = pyabc.ABCSMC(market, priors)
    
    abc_ks.load("sqlite:///"+db_path+"/experiment_1.db", 1)
    abc_war.load("sqlite:///"+db_path+"/experiment_1_war.db", 1)
    
    h_ks = abc_ks.history
    h_war = abc_war.history
    figpath = "../thesis_template/figures/"

    epsilons = pd.concat(
        [h_ks.get_all_populations()["epsilon"], h_war.get_all_populations()["epsilon"]], 
        axis = 1, 
        keys = ["Agg KS statistic", "Sliced-Warsserstein distance"]
        )
    # [:-1]
    
    samples = pd.concat(
        [h_ks.get_all_populations()["samples"], h_war.get_all_populations()["samples"]], 
        axis = 1, 
        keys = ["Agg KS statistic", "Sliced-Warsserstein distance"]
        )
    # [:-1]
    
    samples['generation']=samples.index
    samples = samples.melt(id_vars=['generation'], var_name='distance', value_name='samples') 
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9))
    
    sns.lineplot(ax=axes[0], data=epsilons)
    axes[0].set_xlabel("generation number", fontsize = 12)
    axes[0].set_ylabel("Weighted $\epsilon$", fontsize = 12)
    
    sns.barplot(ax=axes[1], x = 'generation', y="samples", data = samples, 
                hue = 'distance')
    axes[1].set_xlabel("generation number", fontsize = 12)
    axes[1].set_ylabel("# samples", fontsize = 12)
    
    plt.savefig(figpath+'Figure_1_E1.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 22))
    
    for t in range(h_ks.max_t+1):
        df, w = h_ks.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-10,
            xmax=1000,
            x='num_value_ag',
            ax=axes[0,0],
            label=f"PDF t={t+1}",
            refval=params_true,
        )
    axes[0,0].legend()
    axes[0,0].set_xlabel("# value agents", fontsize = 12)
    axes[0,0].set_ylabel(r"$p_{ABC}( \theta_{1} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-10,
            xmax=1000,
            x='num_value_ag',
            ax=axes[0,1],
            label=f"PDF t={t+1}",
            refval=params_true,
        )
    axes[0,1].legend()
    axes[0,1].set_xlabel("# value agents", fontsize = 12)
    axes[0,1].set_ylabel(r"$p_{ABC}( \theta_{1} | y )$", fontsize = 12)
    
    for t in range(h_ks.max_t+1):
        df, w = h_ks.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-10,
            xmax=1000,
            x='num_momentum_ag',
            ax=axes[1,0],
            label=f"PDF t={t+1}",
            refval=params_true,
        )
    axes[1,0].legend()
    axes[1,0].set_xlabel("# momentum agents", fontsize = 12)
    axes[1,0].set_ylabel(r"$p_{ABC}( \theta_{2} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-10,
            xmax=1000,
            x='num_momentum_ag',
            ax=axes[1,1],
            label=f"PDF t={t+1}",
            refval=params_true,
        )
    axes[1,1].legend()
    axes[1,1].set_xlabel("# momentum agents", fontsize = 12)
    axes[1,1].set_ylabel(r"$p_{ABC}( \theta_{2} | y )$", fontsize = 12)
    
    for t in range(h_ks.max_t+1):
        df, w = h_ks.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-10,
            xmax=10000,
            x='num_noise_ag',
            ax=axes[2,0],
            label=f"PDF t={t+1}",
            refval=params_true,
        )
    axes[2,0].legend()
    axes[2,0].set_xlabel("# noise agents", fontsize = 12)
    axes[2,0].set_ylabel(r"$p_{ABC}( \theta_{3} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-10,
            xmax=10000,
            x='num_noise_ag',
            ax=axes[2,1],
            label=f"PDF t={t+1}",
            refval=params_true,
        )
    axes[2,1].legend()
    axes[2,1].set_xlabel("# noise agents", fontsize = 12)
    axes[2,1].set_ylabel(r"$p_{ABC}( \theta_{3} | y )$", fontsize = 12)
    
    for t in range(h_ks.max_t+1):
        df, w = h_ks.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-0.1,
            xmax=0.8,
            x='mm_pov',
            ax=axes[3,0],
            label=f"PDF t={t+1}",
            refval=params_true,
        )
    axes[3,0].legend()
    axes[3,0].set_xlabel("mm POV", fontsize = 12)
    axes[3,0].set_ylabel(r"$p_{ABC}( \theta_{4} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-0.1,
            xmax=0.8,
            x='mm_pov',
            ax=axes[3,1],
            label=f"PDF t={t+1}",
            refval=params_true,
        )
    axes[3,1].legend()
    axes[3,1].set_xlabel("mm POV", fontsize = 12)
    axes[3,1].set_ylabel(r"$p_{ABC}( \theta_{4} | y )$", fontsize = 12)
    
    plt.savefig(figpath+'Figure_2_E1.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    limits = {
        "num_value_ag":(-5, 1000),
        "num_momentum_ag":(-5, 1000),
        "num_noise_ag":(-5, 10000),
        "mm_pov":(-0.1, 0.6)
        }
    
    params_names = ["# value agents", "# momentum agents", "# noise agents", 
                    "mm_pov"]
    
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    pyabc.visualization.plot_kde_matrix_highlevel(
        h_ks, m=0, t=6, limits = limits, refval = params_true, 
        refval_color = "green"
        )
    
    plt.savefig(figpath+'Figure_3_E1.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    pyabc.visualization.plot_kde_matrix_highlevel(
        h_war, m=0, t=6, limits = limits, refval = params_true, 
        refval_color = "green"
        )

    plt.savefig(figpath+'Figure_4_E1.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    sumstats_weighted_war = h_war.get_weighted_sum_stats()
    w_war = sumstats_weighted_war[0]
    sumstats_war = sumstats_weighted_war[1]
    
    min_ret_war = pd.DataFrame(sumstats_war[0]['MinutelyReturns'])
    autocorr_war = pd.DataFrame(sumstats_war[0]['Autocorrelation'])
    volclust_war = pd.DataFrame(sumstats_war[0]['VolatilityClustering'])
    vol_war = pd.DataFrame(sumstats_war[0]['volume_t'])
    imb_war = pd.DataFrame(sumstats_war[0]['imbalance'])
    trades_dist_war = pd.DataFrame(sumstats_war[0]['trades_dist'])

    for i in range(1, len(sumstats_war)):
        min_ret_war[str(i)] = pd.Series(sumstats_war[i]['MinutelyReturns'])
        autocorr_war[str(i)] = pd.Series(sumstats_war[i]['Autocorrelation'])
        volclust_war[str(i)] = pd.Series(sumstats_war[i]['VolatilityClustering'])
        vol_war[str(i)] = pd.Series(sumstats_war[i]['volume_t'])
        imb_war[str(i)] = pd.Series(sumstats_war[i]['imbalance'])
        trades_dist_war[str(i)] = pd.Series(sumstats_war[i]['trades_dist'])
    
    min_ret_war_w = min_ret_war.dot(w_war)
    autocorr_war_w = autocorr_war.dot(w_war)
    volclust_war_w = volclust_war.dot(w_war)
    vol_war_w = vol_war.dot(w_war)
    imb_war_w = imb_war.dot(w_war)
    trades_dist_war_w = trades_dist_war.dot(w_war)
    maxindex = str(w_war.index(max(w_war)))

    df1 = pd.concat([pd.Series(min_ret_war[maxindex]), 
                    pd.Series(data["MinutelyReturns"])], axis = 1)
    df1.colums = ["synthetic", "ground_truth"]
    
    df2 = pd.concat([pd.Series(autocorr_war[maxindex]), 
                    pd.Series(data["Autocorrelation"])], axis = 1)
    df2.colums = ["synthetic", "ground_truth"]
    
    df3 = pd.concat([pd.Series(volclust_war[maxindex]), 
                    pd.Series(data["VolatilityClustering"])], axis = 1)
    df3.colums = ["synthetic", "ground_truth"]
    
    df4 = pd.concat([pd.Series(vol_war_w), 
                    pd.Series(data["volume_t"])], axis = 1)
    df4.colums = ["synthetic", "ground_truth"]
    
    df5 = pd.concat([pd.Series(imb_war[maxindex]), 
                    pd.Series(data["imbalance"])], axis = 1)
    df5.colums = ["synthetic", "ground_truth"]
    
    df6 = pd.concat([pd.Series(trades_dist_war[maxindex]), 
                    pd.Series(data["trades_dist"])], axis = 1)
    df6.colums = ["synthetic", "ground_truth"]
    
    
    sns.set_style("white") 
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    spec = fig.add_gridspec(2, 3)

    ax0 = fig.add_subplot(spec[0, 0])
    annotate_axes(ax0, 'Minutely returns')
    x1 = df1.iloc[:, 0]
    ecdf(x1, x_label='returns', ax = ax0)
    x2 = df1.iloc[:, 1]
    ecdf(x2, ax = ax0)
    plt.legend(['calibrated market', 'rmsc03'])
    
    ax1 = fig.add_subplot(spec[0, 1])
    annotate_axes(ax1, 'Autocorrelation')
    x1 = df2.iloc[:, 0]
    ecdf(x1, x_label='returns auto-correlation', ax = ax1)
    x2 = df2.iloc[:, 1]
    ecdf(x2, ax = ax1)
    plt.legend(['calibrated market', 'rmsc03'])
    
    ax2 = fig.add_subplot(spec[0, 2])
    annotate_axes(ax2, 'Volatility clustering')
    x1 = df3.iloc[:, 0]
    ecdf(x1, x_label='absolute returns correlation', ax = ax2)
    x2 = df3.iloc[:, 1]
    ecdf(x2, ax = ax2)
    plt.legend(['calibrated market', 'rmsc03'])
    
    ax3 = fig.add_subplot(spec[1, 0])
    annotate_axes(ax3, 'Volume')
    x1 = df4.iloc[:, 0]
    ecdf(x1, x_label='volume traded', ax = ax3)
    x2 = df4.iloc[:, 1]
    ecdf(x2, ax = ax3)
    plt.legend(['calibrated market', 'rmsc03'])
    
    ax4 = fig.add_subplot(spec[1, 1])
    annotate_axes(ax4, 'Imbalance')
    x1 = df5.iloc[:, 0]
    ecdf(x1, x_label='%', ax = ax4)
    x2 = df5.iloc[:, 1]
    ecdf(x2, ax = ax4)
    plt.legend(['calibrated market', 'rmsc03'])
    
    ax5 = fig.add_subplot(spec[1, 2])
    annotate_axes(ax5, 'Trades distribution')
    x1 = df6.iloc[:, 0]
    ecdf(x1, x_label='# trades', ax = ax5)
    x2 = df6.iloc[:, 1]
    ecdf(x2, ax = ax5)
    plt.legend(['calibrated market', 'rmsc03'])
    
    plt.savefig(figpath+'Figure_5_E1.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    
    

    
   
        
        
        
        