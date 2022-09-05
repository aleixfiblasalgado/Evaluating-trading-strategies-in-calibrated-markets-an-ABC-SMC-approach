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

from market_simulations import AAPL_10 as market
from plotting_utils import plot_history

from mlxtend.plotting import ecdf

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=get_cmap("tab20").colors) 

if __name__ == "__main__":
    
    fileload = np.load(os.path.join(os.getcwd(), "../data/APPL_ss.npy"), allow_pickle=True)
    data = fileload.item()

    priors = pyabc.Distribution(
        num_value_ag=pyabc.RV('randint', 20, 800),
        num_momentum_ag=pyabc.RV('randint', 10, 500),
        num_noise_ag=pyabc.RV('randint', 10, 8000),
        mm_pov=pyabc.RV("uniform", 0.0001, 0.5),
        kappa=pyabc.RV("uniform", 1e-17, 1e-15),
        fund_vol=pyabc.RV("uniform", 1e-10, 1e-6),
        megashock_lambda_a=pyabc.RV("uniform", 1e-19, 1e-17),
        megashock_mean=pyabc.RV("uniform", 500, 3000),
        megashock_var=pyabc.RV("uniform", 20000, 60000),
    )
    
    db_path = os.path.join(os.getcwd(), "ABC_output")
    
    abc = pyabc.ABCSMC(market, priors)
    
    abc.load("sqlite:///"+db_path+"/experiment_2_war.db", 1)
    
    h_war = abc.history
    figpath = "../thesis_template/figures/"
    
    epsilons = pd.DataFrame(h_war.get_all_populations()["epsilon"])
    
    samples = pd.DataFrame(h_war.get_all_populations()["samples"])
    samples['generation']=samples.index
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 7))
    
    sns.lineplot(ax=axes[0], data=epsilons)
    axes[0].set_xlabel("generation number", fontsize = 14)
    axes[0].set_ylabel("Weighted $\epsilon$", fontsize = 14)
    
    sns.barplot(ax=axes[1], x = 'generation', y="samples", data = samples)
    axes[1].set_xlabel("generation number", fontsize = 14)
    axes[1].set_ylabel("# samples", fontsize = 14)
    
    plt.savefig(figpath+'Figure_1_E2.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
 ##############################################################################   
    
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 22))
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-200,
            xmax=1000,
            x='num_value_ag',
            ax=axes[0,0],
            label=f"PDF t={t+1}"
        )
    axes[0,0].legend()
    axes[0,0].set_xlabel("# value agents", fontsize = 12)
    axes[0,0].set_ylabel(r"$p_{ABC}( \theta_{1} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-100,
            xmax=1000,
            x='num_momentum_ag',
            ax=axes[1,0],
            label=f"PDF t={t+1}",
        )
    axes[1,0].legend()
    axes[1,0].set_xlabel("# momentum agents", fontsize = 12)
    axes[1,0].set_ylabel(r"$p_{ABC}( \theta_{2} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-500,
            xmax=10000,
            x='num_noise_ag',
            ax=axes[2,0],
            label=f"PDF t={t+1}",
        )
    axes[2,0].legend()
    axes[2,0].set_xlabel("# noise agents", fontsize = 12)
    axes[2,0].set_ylabel(r"$p_{ABC}( \theta_{3} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-0.1,
            xmax=0.8,
            x='mm_pov',
            ax=axes[3,0],
            label=f"PDF t={t+1}",
        )
    axes[3,0].legend()
    axes[3,0].set_xlabel("mm POV", fontsize = 12)
    axes[3,0].set_ylabel(r"$p_{ABC}( \theta_{4} | y )$", fontsize = 12)
    
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-8e-16,
            xmax=1e-15,
            x='kappa',
            ax=axes[4,0],
            label=f"PDF t={t+1}",
        )
    axes[4,0].legend()
    axes[4,0].set_xlabel(r"$\kappa$", fontsize = 12)
    axes[4,0].set_ylabel(r"$p_{ABC}( \theta_{5} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-1e-7,
            xmax=1e-6,
            x='fund_vol',
            ax=axes[0,1],
            label=f"PDF t={t+1}",
        )
    axes[0,1].legend()
    axes[0,1].set_xlabel(r"$\sigma$", fontsize = 12)
    axes[0,1].set_ylabel(r"$p_{ABC}( \theta_{6} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-1e-18,
            xmax=2e-17,
            x='megashock_lambda_a',
            ax=axes[1,1],
            label=f"PDF t={t+1}",
        )
    axes[1,1].legend()
    axes[1,1].set_xlabel(r"$\lambda$", fontsize = 12)
    axes[1,1].set_ylabel(r"$p_{ABC}( \theta_{7} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-500,
            xmax=10000,
            x='megashock_mean',
            ax=axes[2,1],
            label=f"PDF t={t+1}",
        )
    axes[2,1].legend()
    axes[2,1].set_xlabel(r"$\mu_{ms}$", fontsize = 12)
    axes[2,1].set_ylabel(r"$p_{ABC}( \theta_{8} | y )$", fontsize = 12)
    
    for t in range(h_war.max_t+1):
        df, w = h_war.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=-1000,
            xmax=100000,
            x='megashock_var',
            ax=axes[3,1],
            label=f"PDF t={t+1}",
        )
    axes[3,1].legend()
    axes[3,1].set_xlabel(r"$var_{ms}$", fontsize = 12)
    axes[3,1].set_ylabel(r"$p_{ABC}( \theta_{9} | y )$", fontsize = 12)
    
    
    axes[4,1].axes.remove()
    
    plt.savefig(figpath+'Figure_2_E2.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
################################################################################ 


    df, w = h_war.get_distribution(m=0, t=6)
    
    limits = {
        "num_value_ag":(-200, 1000),
        "num_momentum_ag":(-100, 1000),
        "num_noise_ag":(-1000, 10000),
        "mm_pov":(-0.1, 0.8)
        }

    pyabc.visualization.plot_kde_matrix(
        df.iloc[:,-4:], 
        w,
        limits = limits
        )
    plt.savefig(figpath+'Figure_3_E2.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    w_list = w.tolist()
    m = max(w_list)
    w_list = [0 if i != m else 1 for i in w_list]
    w_array = np.array(w_list)
    e_w = [1/180]*180
    w_array = np.array(e_w)
    
    limits = {
        "kappa":(-5e-16, 1.2e-15), # 1e-17, 1e-15
        "fund_vol":(-2e-8, 3e-7), # 1e-10, 1e-6
        "megashock_lambda_a":(-5e-18, 1.5e-17), #1e-19, 1e-17
        "megashock_mean":(-500, 5000),
        "megashock_var":(0, 100000)
        }
    
    pyabc.visualization.plot_kde_matrix(
        df.iloc[:,:5], 
        w,
        limits = limits,
        kde=pyabc.transition.MultivariateNormalTransition(scaling=0.3)
        )
    plt.savefig(figpath+'Figure_4_E2.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    limits = {
        "kappa":(-5e-16, 1.2e-15), # 1e-17, 1e-15
        "megashock_lambda_a":(-5e-18, 1.5e-17), #1e-19, 1e-17
        }
    
    pyabc.visualization.plot_kde_matrix(
        df.iloc[:,1:3], 
        w,
        limits = limits
        )
    plt.savefig(figpath+'Figure_6_E2.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    limits = {
        "megashock_mean":(-500, 5000),
        "megashock_var":(0, 100000)
        }
    
    pyabc.visualization.plot_kde_matrix(
        df.iloc[:,[3, 4]], 
        w,
        limits = limits
        )
    plt.savefig(figpath+'Figure_7_E2.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    pyabc.visualization.plot_kde_matrix_highlevel(h_war)
    plt.savefig(figpath+'Figure_8_E2.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
################################################################################ 

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
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    spec = fig.add_gridspec(2, 3)
    
    ax0 = fig.add_subplot(spec[0, 0])
    annotate_axes(ax0, 'Minutely returns')
    x1 = df1.iloc[:, 0]
    ecdf(x1, x_label='returns', ax = ax0)
    x2 = df1.iloc[:, 1]
    ecdf(x2, ax = ax0)
    plt.legend(['summary statistic', 'rmsc03'])
    
    ax1 = fig.add_subplot(spec[0, 1])
    annotate_axes(ax1, 'Autocorrelation')
    x1 = df2.iloc[:, 0]
    ecdf(x1, x_label='auto correlation', ax = ax1)
    x2 = df2.iloc[:, 1]
    ecdf(x2, ax = ax1)
    plt.legend(['summary statistic', 'rmsc03'])
    
    ax2 = fig.add_subplot(spec[0, 2])
    annotate_axes(ax2, 'Volatility clustering')
    x1 = df3.iloc[:, 0]
    ecdf(x1, x_label='absolute returns correlation', ax = ax2)
    x2 = df3.iloc[:, 1]
    ecdf(x2, ax = ax2)
    plt.legend(['summary statistic', 'rmsc03'])
    
    ax3 = fig.add_subplot(spec[1, 0])
    annotate_axes(ax3, 'Volume')
    x1 = df4.iloc[:, 0]
    ecdf(x1, x_label='volume traded', ax = ax3)
    x2 = df4.iloc[:, 1]
    ecdf(x2, ax = ax3)
    plt.legend(['summary statistic', 'rmsc03'])
    
    ax4 = fig.add_subplot(spec[1, 1])
    annotate_axes(ax4, 'Imbalance')
    x1 = df5.iloc[:, 0]
    ecdf(x1, x_label='volume imbalance', ax = ax4)
    x2 = df5.iloc[:, 1]
    ecdf(x2, ax = ax4)
    plt.legend(['summary statistic', 'rmsc03'])
    
    ax5 = fig.add_subplot(spec[1, 2])
    annotate_axes(ax5, 'Trades distribution')
    x1 = df6.iloc[:, 0]
    ecdf(x1, x_label='# trades', ax = ax5)
    x2 = df6.iloc[:, 1]
    ecdf(x2, ax = ax5)
    plt.legend(['summary statistic', 'rmsc03'])
    
    plt.savefig(figpath+'Figure_5_E2.pdf', format='pdf', bbox_inches='tight')
    plt.show()  
        
    
###############################################################################
    
    df, w = h_war.get_distribution(m=0, t=6)
    df_w = df.apply(func=lambda x: np.asarray(x) * np.asarray(w))
    
    sns.set_theme(style="white")
    g = sns.PairGrid(df.iloc[:,:5], diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot, fill = True, thresh=0, levels=100, cmap="rocket")
    g.map_diag(sns.kdeplot, lw=2)
    plt.savefig(figpath+'Figure_9_E2.pdf', format='pdf', bbox_inches='tight')
    plt.show()  
