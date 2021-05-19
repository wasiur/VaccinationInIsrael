from mycolours import *
import os as os
import numpy as np
import scipy as sc
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from numpy.random import RandomState
import seaborn as sns

import pystan
import pickle

rand = RandomState()


from dsacore import DSA 
from DSA_Vaccination_library import DSA2
from mycolours import *


def my_plot_configs():
    plt.style.use('seaborn-paper')
    plt.rcParams["figure.frameon"] = False
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['axes.labelweight'] = 'bold'


def fig_save(fig, Plot_Folder, fname):
    fig.savefig(os.path.join(Plot_Folder, fname), dpi=500)
    fig.savefig(os.path.join(Plot_Folder, fname + "." + 'pdf'),
                format='pdf')
    fig.savefig(os.path.join(Plot_Folder, fname + "." + 'svg'), format='svg')

my_plot_configs()


def euler1d(odefun, endpoints, ic=1.0):
    timepoints = np.linspace(endpoints[0], endpoints[1], 1000)
    stepsize = timepoints[1] - timepoints[0]
    sol = np.zeros(len(timepoints), dtype=np.float64)
    sol[0] = ic
    for i in range(1, len(timepoints)):
        t = timepoints[i-1]
        y = sol[i-1]
        sol[i] = sol[i-1] + np.float64(odefun(t, y)) * stepsize
    return timepoints, sol


def truncate_data_for_DSA(start_date, end_date, df_main):
    n_remove = (df_main.time.max() - end_date).days
    df1 = df_main.drop(df_main.tail(n_remove).index)
    n_remove = (start_date - df_main.time.min()).days
    res = df1.loc[n_remove:]
    return res


def draw_DSA_parms_prior(a_bound=(0.0, 0.5), b_bound=(0.0, 0.5), rho_bound=(0.0, 0.2), nSample=1):
    a_sample = np.random.uniform(low=a_bound[0], high=a_bound[1], size=nSample)
    b_sample = np.random.uniform(low=b_bound[0], high=b_bound[1], size=nSample)
    rho_sample = np.random.uniform(low=rho_bound[0], high=rho_bound[1], size=nSample)
    return a_sample, b_sample, rho_sample


def summary_statistics(trajectories, confidence=0.9):
    upper = 1 - (1-confidence)/2
    lower = (1-confidence)/2

    m_trajectory = np.mean(trajectories, axis=0)
    std_trajectory = np.std(trajectories, axis=0)
    median_trajectory = np.quantile(trajectories, q=0.5, axis=0)
    high_trajectory = np.quantile(trajectories, q=upper, axis=0)
    low_trajectory = np.quantile(trajectories, q=lower, axis=0)
    # high_trajectory = m_trajectory + 1.96*std_trajectory
    # low_trajectory = m_trajectory - 1.96*std_trajectory

    return m_trajectory, std_trajectory, median_trajectory, high_trajectory, low_trajectory


def vaccine_effect_ind_trajectory(df, exit_df,  df_main, my_dict, dsa_parameters, fatality_rate=0.007605500972165701, hospitalization_rate=0.022627883391792468, confidence=0.9, drop=0.5):
    n0 = df_main.cumulative_positive.min() - df_main.daily_positive.iloc[0]
    m0 = df_main.children_cumulative_positive.min(
    ) - df_main.children_daily_positive.iloc[0]
    d0 = df_main.daily_positive.iloc[0]
    d0_Chld = df_main.children_daily_positive.iloc[0]
    h0 = df_main.cumulative_total_hospitalization.min(
    ) - df_main.daily_total_hospital.iloc[0]
    f0 = df_main.cumulative_fatality.min() - df_main.daily_fatality.iloc[0]

    upper = 1 - (1-confidence)/2
    lower = (1-confidence)/2

    trajectories = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)
    cum_cases = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)
    test_pos_probabilities = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.float64)
    chld_trajectories = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)
    chld_cum_cases = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)

    smooth_trajectories = np.zeros((my_dict['a'].size, df_main.time.size))
    smooth_cum_cases = np.zeros((my_dict['a'].size, df_main.time.size))
    smooth_test_pos_probabilities = np.zeros(
        (my_dict['a'].size, df_main.time.size))
    smooth_chld_trajectories = np.zeros((my_dict['a'].size, df_main.time.size))
    smooth_chld_cum_cases = np.zeros((my_dict['a'].size, df_main.time.size))

    hospitalizations = np.zeros((my_dict['a'].size, df_main.time.size))
    cum_hospitalizations = np.zeros((my_dict['a'].size, df_main.time.size))
    fatalities = np.zeros((my_dict['a'].size, df_main.time.size))
    cum_fatalities = np.zeros((my_dict['a'].size, df_main.time.size))

    dsa_trajectories = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_cum_trajectories = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_trajectories_chld = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_cum_trajectories_chld = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)

    dsa_hospitalizations = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_cum_hospitalizations = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_fatalities = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_cum_fatalities = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)

    nDays = df_main.time.size
    time_points = np.arange(nDays)
    n = 9053900 - df_main.cumulative_positive.min()  # Israel's population
    chld_proportion = 0.360496582
    m = np.ceil(n*chld_proportion)

    for i in range(dsa_parameters['a'].size):
        a = dsa_parameters['a'].values[i]
        b = dsa_parameters['b'].values[i]
        rho = dsa_parameters['rho'].values[i]
        #epi = DSA(df=df, a=a, b=b, rho=rho)
        #n = epi.n
        def odefun(t, S): return DSA.poisson_ode_fun(t, S, a=a, b=b, rho=rho)
        t, sol = euler1d(odefun=odefun, endpoints=[0.0, nDays + 1])
        S = interp1d(t, sol)
        dsa_cum_trajectories[i] = np.asarray(
            list(n * (1 - S(x)) + n0 for x in time_points))
        dsa_trajectories[i] = np.append(d0, np.diff(dsa_cum_trajectories[i]))
        #m = np.ceil(n*chld_proportion)
        dsa_cum_trajectories_chld[i] = np.asarray(
            list(m * (1 - S(x)) + m0 for x in time_points))
        dsa_trajectories_chld[i] = np.append(
            d0_Chld, np.diff(dsa_cum_trajectories_chld[i]))

        dsa_hospitalizations[i] = np.multiply(
            dsa_trajectories[i], hospitalization_rate)
        dsa_cum_hospitalizations[i] = np.cumsum(dsa_hospitalizations[i]) + h0
        dsa_fatalities[i] = np.multiply(dsa_trajectories[i], fatality_rate)
        dsa_cum_fatalities[i] = np.cumsum(dsa_fatalities[i]) + f0

    dsa_m, dsa_std, dsa_median, dsa_m_h, dsa_m_l = summary_statistics(
        dsa_cum_trajectories, confidence=confidence)
    dsa_m_daily, dsa_std_daily, dsa_median_daily, dsa_m_daily_h, dsa_m_daily_l = summary_statistics(
        dsa_trajectories, confidence=confidence)
    dsa_m_chld, dsa_std_chld, dsa_median_chld, dsa_m_h_chld, dsa_m_l_chld = summary_statistics(
        dsa_cum_trajectories_chld, confidence=confidence)
    dsa_m_daily_chld, dsa_std_daily_chld, dsa_median_daily_chld, dsa_m_daily_h_chld, dsa_m_daily_l_chld = summary_statistics(
        dsa_trajectories_chld, confidence=confidence)
    dsa_m_hosp, dsa_std_hosp, dsa_median_hosp, dsa_m_h_hosp, dsa_m_l_hosp = summary_statistics(
        dsa_cum_hospitalizations, confidence=confidence)
    dsa_m_daily_hosp, dsa_std_daily_hosp, dsa_median_daily_hosp, dsa_m_daily_h_hosp, dsa_m_daily_l_hosp = summary_statistics(
        dsa_hospitalizations, confidence=confidence)
    dsa_m_fatlty, dsa_std_fatlty, dsa_median_fatlty, dsa_m_h_fatlty, dsa_m_l_fatlty = summary_statistics(
        dsa_cum_fatalities, confidence=confidence)
    dsa_m_daily_fatlty, dsa_std_daily_fatlty, dsa_median_daily_fatlty, dsa_m_daily_h_fatlty, dsa_m_daily_l_fatlty = summary_statistics(
        dsa_fatalities, confidence=confidence)

    for i in range(my_dict['a'].size):
        a = my_dict['a'].values[i]
        b = my_dict['b'].values[i]
        g = my_dict['g'].values[i]
        d = my_dict['d'].values[i]
        l = my_dict['l'].values[i]
        k = my_dict['k'].values[i]
        r_V1 = my_dict['r_V1'].values[i]
        r_V2 = my_dict['r_V2'].values[i]
        r_I = my_dict['r_I'].values[i]
        dsaobj = DSA2(df=exit_df, df_main=df_main, a=a, b=b, g=g,
                      d=0, l=l, k=k, r_V1=0.0, r_V2=0.0, r_I=r_I, drop=drop)
        trajectories[i] = dsaobj.daily_test_pos_prediction()
        cum_cases[i] = np.cumsum(trajectories[i]) + n0
        test_pos_probabilities[i] = dsaobj.daily_test_pos_probabilities()
        chld_trajectories[i] = dsaobj.children_daily_test_pos_prediction()
        chld_cum_cases[i] = np.cumsum(chld_trajectories[i]) + m0
        smooth_trajectories[i] = dsaobj.daily_test_pos_prediction_smooth()
        #smooth_trajectories[i] = pd.DataFrame(trajectories[i]).rolling(7, min_periods=1).mean().values
        smooth_cum_cases[i] = np.cumsum(smooth_trajectories[i]) + n0
        #smooth_chld_trajectories[i] = pd.DataFrame(dsaobj.children_daily_test_pos_prediction()).rolling(7, min_periods=1).mean().values
        smooth_chld_trajectories[i] = dsaobj.children_daily_test_pos_prediction_smooth(
        )
        smooth_chld_cum_cases[i] = np.cumsum(smooth_chld_trajectories[i]) + m0

        hospitalizations[i] = np.multiply(
            smooth_trajectories[i], hospitalization_rate)
        cum_hospitalizations[i] = np.cumsum(hospitalizations[i]) + h0
        fatalities[i] = np.multiply(smooth_trajectories[i], fatality_rate)
        cum_fatalities[i] = np.cumsum(fatalities[i]) + f0

    m_trajectory, std_trajectory, median_trajectory, high_trajectory, low_trajectory = summary_statistics(
        trajectories, confidence=confidence)
    m_cum_cases, std_cum_cases, median_cum_cases, high_cum_cases, low_cum_cases = summary_statistics(
        cum_cases, confidence=confidence)
    smooth_m_trajectory, smooth_std_trajectory, smooth_median_trajectory, smooth_high_trajectory, smooth_low_trajectory = summary_statistics(
        smooth_trajectories, confidence=confidence)
    smooth_m_cum_cases, smooth_std_cum_cases, smooth_median_cum_cases, smooth_high_cum_cases, smooth_low_cum_cases = summary_statistics(
        smooth_cum_cases, confidence=confidence)
    m_prob, std_prob, median_prob, high_prob, low_prob = summary_statistics(
        test_pos_probabilities, confidence=confidence)
    m_trajectory_chld, std_trajectory_chld, median_trajectory_chld, high_trajectory_chld, low_trajectory_chld = summary_statistics(
        chld_trajectories, confidence=confidence)
    m_cum_cases_chld, std_cum_cases_chld, median_cum_cases_chld, high_cum_cases_chld, low_cum_cases_chld = summary_statistics(
        chld_cum_cases, confidence=confidence)
    smooth_m_trajectory_chld, smooth_std_trajectory_chld, smooth_median_trajectory_chld, smooth_high_trajectory_chld, smooth_low_trajectory_chld = summary_statistics(
        smooth_chld_trajectories, confidence=confidence)
    smooth_m_cum_cases_chld, smooth_std_cum_cases_chld, smooth_median_cum_cases_chld, smooth_high_cum_cases_chld, smooth_low_cum_cases_chld = summary_statistics(
        smooth_chld_cum_cases, confidence=confidence)
    m_daily_hosp, std_daily_hosp, median_daily_hosp, high_daily_hosp, low_daily_hosp = summary_statistics(
        hospitalizations, confidence=confidence)
    m_cum_hosp, std_cum_hosp, median_cum_hosp, high_cum_hosp, low_cum_hosp = summary_statistics(
        cum_hospitalizations, confidence=confidence)
    m_cum_fatlty, std_cum_fatlty, median_cum_fatlty, high_cum_fatlty, low_cum_fatlty = summary_statistics(
        cum_fatalities, confidence=confidence)
    m_daily_fatlty, std_daily_fatlty, median_daily_fatlty, high_daily_fatlty, low_daily_fatlty = summary_statistics(
        fatalities, confidence=confidence)

    my_plot_configs()

    fig_trajectory = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
        # plt.plot(df_main.time.values,
                #  trajectories[i], '-', color=bluegreys['bluegrey1'].get_rgb(), alpha=0.5)
    plt.fill_between(df_main.time.values, low_trajectory, high_trajectory,
                     alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l,
                     dsa_m_daily_h, alpha=.5, color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_trajectory, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    # sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    ax.set_ylim([0, None])
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_trajectory_smooth = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, smooth_low_trajectory,
                     smooth_high_trajectory, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l,
                     dsa_m_daily_h, alpha=.5, color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, smooth_m_trajectory, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0, None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_cum_cases = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_cases, high_cum_cases,
                     alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_l, dsa_m_h,
                     alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_cases, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0, None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()

    fig_prob = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_prob, high_prob,
                     alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_pct_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_prob, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: no vaccination')
    plt.legend(handles=[l1, l2])
    plt.xlabel('Date')
    plt.ylabel('Probability of testing positive')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0, None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()

    fig_trajectory_chld = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_trajectory_chld,
                     high_trajectory_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l_chld,
                     dsa_m_daily_h_chld, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.children_daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_trajectory_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily_chld, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0, None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_trajectory_chld_smooth = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, smooth_low_trajectory_chld,
                     smooth_high_trajectory_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l_chld,
                     dsa_m_daily_h_chld, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.children_daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, smooth_m_trajectory_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily_chld, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0, None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_cum_cases_chld = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_cases_chld,
                     high_cum_cases_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_l_chld,
                     dsa_m_h_chld, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.children_cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_cases_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_chld, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0, None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()

    fig_hosp = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_daily_hosp,
                     high_daily_hosp, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l_hosp,
                     dsa_m_daily_h_hosp, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.daily_total_hospital.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_daily_hosp, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily_hosp, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily hospitalization')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0, None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_cum_hosp = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_hosp,
                     high_cum_hosp, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_l_hosp,
                     dsa_m_h_hosp, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.cumulative_total_hospitalization.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_hosp, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_hosp, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative hospitalization')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0, None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()

    fig_fatality = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_daily_fatlty,
                     high_daily_fatlty, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l_fatlty,
                     dsa_m_daily_h_fatlty, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.daily_fatality.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_daily_fatlty, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily_fatlty, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily fatality')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0, None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_cum_fatality = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_fatlty,
                     high_cum_fatlty, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_l_fatlty,
                     dsa_m_h_fatlty, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.cumulative_fatality.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_fatlty, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_fatlty, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative fatality')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0, None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()

    counterfactuals_dict = {}
    counterfactuals_dict['Date'] = df_main.time.values
    counterfactuals_dict['True_Daily_Cases'] = df_main.daily_positive.values
    counterfactuals_dict['True_Cumulative_Cases'] = df_main.cumulative_positive.values
    counterfactuals_dict['True_Daily_Pct_Positive'] = df_main.daily_pct_positive.values
    counterfactuals_dict['True_Daily_Cases_Chld'] = df_main.children_daily_positive.values
    counterfactuals_dict['True_Cumulative_Cases_Chld'] = df_main.children_cumulative_positive.values
    counterfactuals_dict['True_Daily_Pct_Positive_Chld'] = df_main.children_daily_pct_positive.values
    counterfactuals_dict['True_Daily_Hospitalization'] = df_main.daily_total_hospital.values
    counterfactuals_dict['True_Cumulative_Hospitalization'] = df_main.cumulative_total_hospitalization.values
    counterfactuals_dict['True_Daily_Fatality'] = df_main.daily_fatality.values
    counterfactuals_dict['True_Cumulative_Fatality'] = df_main.cumulative_fatality.values

    counterfactuals_dict['Mean_DSA_Cumulative'] = dsa_m
    counterfactuals_dict['Low_DSA_Cumulative'] = dsa_m_l
    counterfactuals_dict['High_DSA_Cumulative'] = dsa_m_h

    counterfactuals_dict['Mean_DSA_Cumulative_Chld'] = dsa_m_chld
    counterfactuals_dict['Low_DSA_Cumulative_Chld'] = dsa_m_l_chld
    counterfactuals_dict['High_DSA_Cumulative_Chld'] = dsa_m_h_chld

    counterfactuals_dict['Mean_Daily_Cases_NoVac'] = m_trajectory
    counterfactuals_dict['Median_Daily_Cases_NoVac'] = median_trajectory
    counterfactuals_dict['High_Daily_Cases_NoVac'] = high_trajectory
    counterfactuals_dict['Low_Daily_Cases_NoVac'] = low_trajectory
    counterfactuals_dict['Mean_Cumulative_Cases_NoVac'] = m_cum_cases
    counterfactuals_dict['High_Cumulative_Cases_NoVac'] = high_cum_cases
    counterfactuals_dict['Low_Cumulative_Cases_NoVac'] = low_cum_cases

    counterfactuals_dict['Mean_Cumulative_Cases_Children'] = m_cum_cases_chld
    counterfactuals_dict['Median_Cumulative_Cases_Children'] = median_cum_cases_chld
    counterfactuals_dict['High_Cumulative_Cases_Children'] = high_cum_cases_chld
    counterfactuals_dict['Low_Cumulative_Cases_Children'] = low_cum_cases_chld
    counterfactuals_dict['Mean_Daily_Children_NoVac'] = m_trajectory_chld
    counterfactuals_dict['Median_Daily_Children_NoVac'] = median_trajectory_chld
    counterfactuals_dict['High_Daily_Children_NoVac'] = high_trajectory_chld
    counterfactuals_dict['Low_Daily_Children_NoVac'] = low_trajectory_chld
    counterfactuals_dict['Mean_Cumulative_Cases_Children_NoVac'] = m_cum_cases_chld
    counterfactuals_dict['Median_Cumulative_Cases_Children_NoVac'] = median_cum_cases_chld
    counterfactuals_dict['High_Cumulative_Cases_Children_NoVac'] = high_cum_cases_chld
    counterfactuals_dict['Low_Cumulative_Cases_Children_NoVac'] = low_cum_cases_chld

    counterfactuals_dict['Mean_Daily_Hospitalization'] = m_daily_hosp
    counterfactuals_dict['High_Daily_Hospitalization'] = high_daily_hosp
    counterfactuals_dict['Low_Daily_Hospitalization'] = low_daily_hosp
    counterfactuals_dict['Mean_Cumulative_Hospitalization'] = m_cum_hosp
    counterfactuals_dict['High_Cumulative_Hospitalization'] = high_cum_hosp
    counterfactuals_dict['Low_Cumulative_Hospitalization'] = low_cum_hosp

    counterfactuals_dict['Mean_Daily_Fatality'] = m_daily_fatlty
    counterfactuals_dict['High_Daily_Fatality'] = high_daily_fatlty
    counterfactuals_dict['Low_Daily_Fatality'] = low_daily_fatlty
    counterfactuals_dict['Mean_Cumulative_Fatality'] = m_cum_fatlty
    counterfactuals_dict['High_Cumulative_Fatality'] = high_cum_fatlty
    counterfactuals_dict['Low_Cumulative_Fatality'] = low_cum_fatlty

    counterfactuals_dict['Mean_DSA_Daily_Hospitalization'] = dsa_m_daily_hosp
    counterfactuals_dict['High_DSA_Daily_Hospitalization'] = dsa_m_daily_h_hosp
    counterfactuals_dict['Low_DSA_Daily_Hospitalization'] = dsa_m_daily_l_hosp
    counterfactuals_dict['Mean_DSA_Cumulative_Hospitalization'] = dsa_m_hosp
    counterfactuals_dict['High_DSA_Cumulative_Hospitalization'] = dsa_m_h_hosp
    counterfactuals_dict['Low_DSA_Cumulative_Hospitalization'] = dsa_m_l_hosp

    counterfactuals_dict['Mean_DSA_Daily_Fatality'] = dsa_m_daily_fatlty
    counterfactuals_dict['High_DSA_Daily_Fatality'] = dsa_m_daily_h_fatlty
    counterfactuals_dict['Low_DSA_Daily_Fatality'] = dsa_m_daily_l_fatlty
    counterfactuals_dict['Mean_DSA_Cumulative_Fatality'] = dsa_m_fatlty
    counterfactuals_dict['High_DSA_Cumulative_Fatality'] = dsa_m_h_fatlty
    counterfactuals_dict['Low_DSA_Cumulative_Fatality'] = dsa_m_l_fatlty

    counterfactuals_dict['Diff_NoVac_Mean'] = counterfactuals_dict['Mean_Cumulative_Cases_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_NoVac_Low'] = counterfactuals_dict['Low_Cumulative_Cases_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_NoVac_High'] = counterfactuals_dict['High_Cumulative_Cases_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases']

    counterfactuals_dict['Diff_NoVac_Chld_Mean'] = counterfactuals_dict['Mean_Cumulative_Cases_Children_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_NoVac_Chld_Low'] = counterfactuals_dict['Low_Cumulative_Cases_Children_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_NoVac_Chld_High'] = counterfactuals_dict['High_Cumulative_Cases_Children_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']

    counterfactuals_dict['Diff_DSA_Mean'] = counterfactuals_dict['Mean_DSA_Cumulative'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_DSA_Low'] = counterfactuals_dict['Low_DSA_Cumulative'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_DSA_High'] = counterfactuals_dict['High_DSA_Cumulative'] - \
        counterfactuals_dict['True_Cumulative_Cases']

    counterfactuals_dict['Diff_DSA_Mean_Chld'] = counterfactuals_dict['Mean_DSA_Cumulative_Chld'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_DSA_Low_Chld'] = counterfactuals_dict['Low_DSA_Cumulative_Chld'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_DSA_High_Chld'] = counterfactuals_dict['High_DSA_Cumulative_Chld'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']

    counterfactuals_dict['Diff_Mean_Hospitalization'] = counterfactuals_dict['True_Cumulative_Hospitalization'] - \
        counterfactuals_dict['Mean_Cumulative_Hospitalization']
    counterfactuals_dict['Diff_Low_Hospitalization'] = counterfactuals_dict['True_Cumulative_Hospitalization'] - \
        counterfactuals_dict['Low_Cumulative_Hospitalization']
    counterfactuals_dict['Diff_High_Hospitalization'] = counterfactuals_dict['True_Cumulative_Hospitalization'] - \
        counterfactuals_dict['High_Cumulative_Hospitalization']

    counterfactuals_dict['Diff_DSA_Mean_Hospitalization'] = counterfactuals_dict['True_Cumulative_Hospitalization'] - \
        counterfactuals_dict['Mean_DSA_Cumulative_Hospitalization']
    counterfactuals_dict['Diff_DSA_Low_Hospitalization'] = counterfactuals_dict['True_Cumulative_Hospitalization'] - \
        counterfactuals_dict['Low_DSA_Cumulative_Hospitalization']
    counterfactuals_dict['Diff_DSA_High_Hospitalization'] = counterfactuals_dict['True_Cumulative_Hospitalization'] - \
        counterfactuals_dict['High_DSA_Cumulative_Hospitalization']

    counterfactuals_dict['Diff_Mean_Fatality'] = counterfactuals_dict['True_Cumulative_Fatality'] - \
        counterfactuals_dict['Mean_Cumulative_Fatality']
    counterfactuals_dict['Diff_Low_Fatality'] = counterfactuals_dict['True_Cumulative_Fatality'] - \
        counterfactuals_dict['Low_Cumulative_Fatality']
    counterfactuals_dict['Diff_High_Fatality'] = counterfactuals_dict['True_Cumulative_Fatality'] - \
        counterfactuals_dict['High_Cumulative_Fatality']

    counterfactuals_dict['Diff_DSA_Mean_Fatality'] = counterfactuals_dict['True_Cumulative_Fatality'] - \
        counterfactuals_dict['Mean_DSA_Cumulative_Fatality']
    counterfactuals_dict['Diff_DSA_Low_Fatality'] = counterfactuals_dict['True_Cumulative_Fatality'] - \
        counterfactuals_dict['Low_DSA_Cumulative_Fatality']
    counterfactuals_dict['Diff_DSA_High_Fatality'] = counterfactuals_dict['True_Cumulative_Fatality'] - \
        counterfactuals_dict['High_DSA_Cumulative_Fatality']

    counterfactuals_dict = pd.DataFrame(counterfactuals_dict)

    return counterfactuals_dict, fig_trajectory, fig_cum_cases, fig_prob, fig_trajectory_chld, fig_cum_cases_chld, fig_trajectory_smooth, fig_trajectory_chld_smooth, fig_hosp, fig_cum_hosp, fig_fatality, fig_cum_fatality




def vaccine_effect(df, exit_df,  df_main, my_dict, dsa_parameters, fatality_rate=0.007605500972165701, hospitalization_rate=0.022627883391792468, confidence=0.9, drop=0.5):
    n0 = df_main.cumulative_positive.min() - df_main.daily_positive.iloc[0]
    m0 = df_main.children_cumulative_positive.min(
    ) - df_main.children_daily_positive.iloc[0]
    d0 = df_main.daily_positive.iloc[0]
    d0_Chld = df_main.children_daily_positive.iloc[0]
    h0 = df_main.cumulative_total_hospitalization.min() - df_main.daily_total_hospital.iloc[0]
    f0 = df_main.cumulative_fatality.min() - df_main.daily_fatality.iloc[0]

    upper = 1 - (1-confidence)/2
    lower = (1-confidence)/2

    trajectories = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)
    cum_cases = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)
    test_pos_probabilities = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.float64)
    chld_trajectories = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)
    chld_cum_cases = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)

    smooth_trajectories = np.zeros((my_dict['a'].size, df_main.time.size))
    smooth_cum_cases = np.zeros((my_dict['a'].size, df_main.time.size))
    smooth_test_pos_probabilities = np.zeros(
        (my_dict['a'].size, df_main.time.size))
    smooth_chld_trajectories = np.zeros((my_dict['a'].size, df_main.time.size))
    smooth_chld_cum_cases = np.zeros((my_dict['a'].size, df_main.time.size))

    hospitalizations = np.zeros((my_dict['a'].size, df_main.time.size))
    cum_hospitalizations = np.zeros((my_dict['a'].size, df_main.time.size))
    fatalities=np.zeros((my_dict['a'].size, df_main.time.size))
    cum_fatalities = np.zeros((my_dict['a'].size, df_main.time.size))

    dsa_trajectories = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_cum_trajectories = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_trajectories_chld = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_cum_trajectories_chld = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)

    dsa_hospitalizations = np.zeros((dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_cum_hospitalizations = np.zeros((dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_fatalities = np.zeros((dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_cum_fatalities = np.zeros((dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)

    nDays = df_main.time.size
    time_points = np.arange(nDays)
    n = 9053900 - df_main.cumulative_positive.min()  # Israel's population
    chld_proportion = 0.360496582
    m = np.ceil(n*chld_proportion)

    for i in range(dsa_parameters['a'].size):
        a = dsa_parameters['a'].values[i]
        b = dsa_parameters['b'].values[i]
        rho = dsa_parameters['rho'].values[i]
        #epi = DSA(df=df, a=a, b=b, rho=rho)
        #n = epi.n
        def odefun(t, S): return DSA.poisson_ode_fun(t, S, a=a, b=b, rho=rho)
        t, sol = euler1d(odefun=odefun, endpoints=[0.0, nDays + 1])
        S = interp1d(t, sol)
        dsa_cum_trajectories[i] = np.asarray(
            list(n * (1 - S(x)) + n0 for x in time_points))
        dsa_trajectories[i] = np.append(d0, np.diff(dsa_cum_trajectories[i]))
        #m = np.ceil(n*chld_proportion)
        dsa_cum_trajectories_chld[i] = np.asarray(
            list(m * (1 - S(x)) + m0 for x in time_points))
        dsa_trajectories_chld[i] = np.append(
            d0_Chld, np.diff(dsa_cum_trajectories_chld[i]))

        dsa_hospitalizations[i]= np.multiply(dsa_trajectories[i], hospitalization_rate)
        dsa_cum_hospitalizations[i] = np.cumsum(dsa_hospitalizations[i]) + h0
        dsa_fatalities[i] = np.multiply(dsa_trajectories[i], fatality_rate)
        dsa_cum_fatalities[i]=np.cumsum(dsa_fatalities[i]) + f0

    dsa_m, dsa_std, dsa_median, dsa_m_h, dsa_m_l = summary_statistics(dsa_cum_trajectories, confidence=confidence)
    dsa_m_daily, dsa_std_daily, dsa_median_daily, dsa_m_daily_h, dsa_m_daily_l = summary_statistics(dsa_trajectories, confidence=confidence)
    dsa_m_chld, dsa_std_chld, dsa_median_chld, dsa_m_h_chld, dsa_m_l_chld = summary_statistics(dsa_cum_trajectories_chld, confidence=confidence)
    dsa_m_daily_chld, dsa_std_daily_chld, dsa_median_daily_chld, dsa_m_daily_h_chld, dsa_m_daily_l_chld = summary_statistics(dsa_trajectories_chld, confidence=confidence)
    dsa_m_hosp, dsa_std_hosp, dsa_median_hosp, dsa_m_h_hosp, dsa_m_l_hosp = summary_statistics(dsa_cum_hospitalizations, confidence=confidence)
    dsa_m_daily_hosp, dsa_std_daily_hosp, dsa_median_daily_hosp, dsa_m_daily_h_hosp, dsa_m_daily_l_hosp = summary_statistics(dsa_hospitalizations, confidence=confidence)
    dsa_m_fatlty, dsa_std_fatlty, dsa_median_fatlty, dsa_m_h_fatlty, dsa_m_l_fatlty = summary_statistics(dsa_cum_fatalities, confidence=confidence)
    dsa_m_daily_fatlty, dsa_std_daily_fatlty, dsa_median_daily_fatlty, dsa_m_daily_h_fatlty, dsa_m_daily_l_fatlty = summary_statistics(dsa_fatalities, confidence=confidence)
    
    for i in range(my_dict['a'].size):
        a = my_dict['a'].values[i]
        b = my_dict['b'].values[i]
        g = my_dict['g'].values[i]
        d = my_dict['d'].values[i]
        l = my_dict['l'].values[i]
        k = my_dict['k'].values[i]
        r_V1 = my_dict['r_V1'].values[i]
        r_V2 = my_dict['r_V2'].values[i]
        r_I = my_dict['r_I'].values[i]
        dsaobj = DSA2(df=exit_df, df_main=df_main, a=a, b=b, g=g,
                      d=0, l=l, k=k, r_V1=0.0, r_V2=0.0, r_I=r_I, drop=drop)
        trajectories[i] = dsaobj.daily_test_pos_prediction()
        cum_cases[i] = np.cumsum(trajectories[i]) + n0
        test_pos_probabilities[i] = dsaobj.daily_test_pos_probabilities()
        chld_trajectories[i] = dsaobj.children_daily_test_pos_prediction()
        chld_cum_cases[i] = np.cumsum(chld_trajectories[i]) + m0
        smooth_trajectories[i] = dsaobj.daily_test_pos_prediction_smooth()
        #smooth_trajectories[i] = pd.DataFrame(trajectories[i]).rolling(7, min_periods=1).mean().values
        smooth_cum_cases[i] = np.cumsum(smooth_trajectories[i]) + n0
        #smooth_chld_trajectories[i] = pd.DataFrame(dsaobj.children_daily_test_pos_prediction()).rolling(7, min_periods=1).mean().values
        smooth_chld_trajectories[i] = dsaobj.children_daily_test_pos_prediction_smooth()
        smooth_chld_cum_cases[i] = np.cumsum(smooth_chld_trajectories[i]) + m0

        hospitalizations[i]= np.multiply(smooth_trajectories[i], hospitalization_rate)
        cum_hospitalizations[i] = np.cumsum(hospitalizations[i]) + h0
        fatalities[i] = np.multiply(smooth_trajectories[i], fatality_rate)
        cum_fatalities[i] = np.cumsum(fatalities[i]) + f0

    m_trajectory, std_trajectory, median_trajectory, high_trajectory, low_trajectory = summary_statistics(trajectories, confidence=confidence)
    m_cum_cases, std_cum_cases, median_cum_cases, high_cum_cases, low_cum_cases = summary_statistics(cum_cases, confidence=confidence)
    smooth_m_trajectory, smooth_std_trajectory, smooth_median_trajectory, smooth_high_trajectory, smooth_low_trajectory = summary_statistics(smooth_trajectories, confidence=confidence)
    smooth_m_cum_cases, smooth_std_cum_cases, smooth_median_cum_cases, smooth_high_cum_cases, smooth_low_cum_cases = summary_statistics(smooth_cum_cases, confidence=confidence)
    m_prob, std_prob, median_prob, high_prob, low_prob = summary_statistics(test_pos_probabilities, confidence=confidence)
    m_trajectory_chld, std_trajectory_chld, median_trajectory_chld, high_trajectory_chld, low_trajectory_chld = summary_statistics(chld_trajectories, confidence=confidence)
    m_cum_cases_chld, std_cum_cases_chld, median_cum_cases_chld, high_cum_cases_chld, low_cum_cases_chld=summary_statistics(chld_cum_cases, confidence=confidence)
    smooth_m_trajectory_chld, smooth_std_trajectory_chld, smooth_median_trajectory_chld, smooth_high_trajectory_chld, smooth_low_trajectory_chld = summary_statistics(smooth_chld_trajectories, confidence=confidence)
    smooth_m_cum_cases_chld, smooth_std_cum_cases_chld, smooth_median_cum_cases_chld, smooth_high_cum_cases_chld, smooth_low_cum_cases_chld = summary_statistics(smooth_chld_cum_cases, confidence=confidence)
    m_daily_hosp, std_daily_hosp, median_daily_hosp, high_daily_hosp, low_daily_hosp = summary_statistics(hospitalizations, confidence=confidence)
    m_cum_hosp, std_cum_hosp, median_cum_hosp, high_cum_hosp, low_cum_hosp = summary_statistics(cum_hospitalizations, confidence=confidence)
    m_cum_fatlty, std_cum_fatlty, median_cum_fatlty, high_cum_fatlty, low_cum_fatlty = summary_statistics(cum_fatalities, confidence=confidence)
    m_daily_fatlty, std_daily_fatlty, median_daily_fatlty, high_daily_fatlty, low_daily_fatlty = summary_statistics(fatalities, confidence=confidence)


    my_plot_configs()

    fig_trajectory = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_trajectory, high_trajectory,
                     alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l,
                     dsa_m_daily_h, alpha=.5, color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_trajectory, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    # sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    ax.set_ylim([0,None])
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)


    fig_trajectory_smooth = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, smooth_low_trajectory,
                     smooth_high_trajectory, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l,
                     dsa_m_daily_h, alpha=.5, color=greys['grey1'].get_rgb())
    # l1, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
    #                lw=2, label='Actual')
    l1, = plt.plot(df_main.time.values, df_main.daily_positive.rolling(7, min_periods=1).mean().values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, smooth_m_trajectory, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_cum_cases = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_cases, high_cum_cases,
                     alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_l, dsa_m_h,
                     alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_cases, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()

    fig_prob = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_prob, high_prob,
                     alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_pct_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_prob, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: no vaccination')
    plt.legend(handles=[l1, l2])
    plt.xlabel('Date')
    plt.ylabel('Probability of testing positive')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()

    fig_trajectory_chld = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_trajectory_chld,
                     high_trajectory_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l_chld,
                     dsa_m_daily_h_chld, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.children_daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_trajectory_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily_chld, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_trajectory_chld_smooth = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, smooth_low_trajectory_chld,
                     smooth_high_trajectory_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l_chld,
                     dsa_m_daily_h_chld, alpha=.5, color=greys['grey1'].get_rgb())

    # l1, = plt.plot(df_main.time.values, df_main.children_daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
    #                lw=2, label='Actual')
    l1, = plt.plot(df_main.time.values, df_main.children_daily_positive.rolling(7, min_periods=1).mean().values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, smooth_m_trajectory_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily_chld, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_cum_cases_chld = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_cases_chld,
                     high_cum_cases_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_l_chld,
                     dsa_m_h_chld, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.children_cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_cases_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_chld, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()



    fig_hosp = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_daily_hosp,
                        high_daily_hosp, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l_hosp,
                        dsa_m_daily_h_hosp, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.daily_total_hospital.values, '-', color=maroons['maroon3'].get_rgb(),
                    lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_daily_hosp, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                    lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily_hosp, '-.', color=greys['grey4'].get_rgb(),
                    lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily hospitalization')
    sns.despine()
    ax=plt.gca()
    ax.set_ylim([0, None])
    date_form=DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_cum_hosp = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_hosp,
                        high_cum_hosp, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_l_hosp,
                        dsa_m_h_hosp, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.cumulative_total_hospitalization.values, '-', color=maroons['maroon3'].get_rgb(),
                    lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_hosp, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                    lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_hosp, '-.', color=greys['grey4'].get_rgb(),
                    lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative hospitalization')
    sns.despine()
    ax=plt.gca()
    ax.set_ylim([0, None])
    date_form=DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()



    fig_fatality = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_daily_fatlty,
                        high_daily_fatlty, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l_fatlty,
                        dsa_m_daily_h_fatlty, alpha=.5, color=greys['grey1'].get_rgb())

    l1,= plt.plot(df_main.time.values, df_main.daily_fatality.values, '-', color=maroons['maroon3'].get_rgb(),
                    lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_daily_fatlty, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                    lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily_fatlty, '-.', color=greys['grey4'].get_rgb(),
                    lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily fatality')
    sns.despine()
    ax=plt.gca()
    ax.set_ylim([0, None])
    date_form=DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_cum_fatality = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_fatlty,
                        high_cum_fatlty, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_l_fatlty,
                        dsa_m_h_fatlty, alpha=.5, color=greys['grey1'].get_rgb())

    l1,= plt.plot(df_main.time.values, df_main.cumulative_fatality.values, '-', color=maroons['maroon3'].get_rgb(),
                    lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_fatlty, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                    lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_fatlty, '-.', color=greys['grey4'].get_rgb(),
                    lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative fatality')
    sns.despine()
    ax=plt.gca()
    ax.set_ylim([0, None])
    date_form=DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()



    counterfactuals_dict = {}
    counterfactuals_dict['Date'] = df_main.time.values
    counterfactuals_dict['True_Daily_Cases'] = df_main.daily_positive.values
    counterfactuals_dict['True_Cumulative_Cases'] = df_main.cumulative_positive.values
    counterfactuals_dict['True_Daily_Pct_Positive'] = df_main.daily_pct_positive.values
    counterfactuals_dict['True_Daily_Cases_Chld'] = df_main.children_daily_positive.values
    counterfactuals_dict['True_Cumulative_Cases_Chld'] = df_main.children_cumulative_positive.values
    counterfactuals_dict['True_Daily_Pct_Positive_Chld'] = df_main.children_daily_pct_positive.values
    counterfactuals_dict['True_Daily_Hospitalization'] = df_main.daily_total_hospital.values
    counterfactuals_dict['True_Cumulative_Hospitalization'] = df_main.cumulative_total_hospitalization.values
    counterfactuals_dict['True_Daily_Fatality'] = df_main.daily_fatality.values
    counterfactuals_dict['True_Cumulative_Fatality'] = df_main.cumulative_fatality.values

    counterfactuals_dict['Mean_DSA_Cumulative'] = dsa_m
    counterfactuals_dict['Low_DSA_Cumulative'] = dsa_m_l
    counterfactuals_dict['High_DSA_Cumulative'] = dsa_m_h

    counterfactuals_dict['Mean_DSA_Cumulative_Chld'] = dsa_m_chld
    counterfactuals_dict['Low_DSA_Cumulative_Chld'] = dsa_m_l_chld
    counterfactuals_dict['High_DSA_Cumulative_Chld'] = dsa_m_h_chld

    counterfactuals_dict['Mean_Daily_Cases_NoVac'] = m_trajectory
    counterfactuals_dict['Median_Daily_Cases_NoVac'] = median_trajectory
    counterfactuals_dict['High_Daily_Cases_NoVac'] = high_trajectory
    counterfactuals_dict['Low_Daily_Cases_NoVac'] = low_trajectory
    counterfactuals_dict['Mean_Cumulative_Cases_NoVac'] = m_cum_cases
    counterfactuals_dict['High_Cumulative_Cases_NoVac'] = high_cum_cases
    counterfactuals_dict['Low_Cumulative_Cases_NoVac'] = low_cum_cases

    counterfactuals_dict['Mean_Cumulative_Cases_Children'] = m_cum_cases_chld
    counterfactuals_dict['Median_Cumulative_Cases_Children'] = median_cum_cases_chld
    counterfactuals_dict['High_Cumulative_Cases_Children'] = high_cum_cases_chld
    counterfactuals_dict['Low_Cumulative_Cases_Children'] = low_cum_cases_chld
    counterfactuals_dict['Mean_Daily_Children_NoVac'] = m_trajectory_chld
    counterfactuals_dict['Median_Daily_Children_NoVac'] = median_trajectory_chld
    counterfactuals_dict['High_Daily_Children_NoVac'] = high_trajectory_chld
    counterfactuals_dict['Low_Daily_Children_NoVac'] = low_trajectory_chld
    counterfactuals_dict['Mean_Cumulative_Cases_Children_NoVac'] = m_cum_cases_chld
    counterfactuals_dict['Median_Cumulative_Cases_Children_NoVac'] = median_cum_cases_chld
    counterfactuals_dict['High_Cumulative_Cases_Children_NoVac'] = high_cum_cases_chld
    counterfactuals_dict['Low_Cumulative_Cases_Children_NoVac'] = low_cum_cases_chld

    counterfactuals_dict['Mean_Daily_Hospitalization'] = m_daily_hosp
    counterfactuals_dict['High_Daily_Hospitalization'] = high_daily_hosp
    counterfactuals_dict['Low_Daily_Hospitalization'] = low_daily_hosp
    counterfactuals_dict['Mean_Cumulative_Hospitalization'] = m_cum_hosp
    counterfactuals_dict['High_Cumulative_Hospitalization'] = high_cum_hosp
    counterfactuals_dict['Low_Cumulative_Hospitalization'] = low_cum_hosp

    counterfactuals_dict['Mean_Daily_Fatality'] = m_daily_fatlty
    counterfactuals_dict['High_Daily_Fatality'] = high_daily_fatlty
    counterfactuals_dict['Low_Daily_Fatality'] = low_daily_fatlty
    counterfactuals_dict['Mean_Cumulative_Fatality'] = m_cum_fatlty
    counterfactuals_dict['High_Cumulative_Fatality'] = high_cum_fatlty
    counterfactuals_dict['Low_Cumulative_Fatality'] = low_cum_fatlty

    counterfactuals_dict['Mean_DSA_Daily_Hospitalization'] = dsa_m_daily_hosp
    counterfactuals_dict['High_DSA_Daily_Hospitalization'] = dsa_m_daily_h_hosp
    counterfactuals_dict['Low_DSA_Daily_Hospitalization'] = dsa_m_daily_l_hosp
    counterfactuals_dict['Mean_DSA_Cumulative_Hospitalization'] = dsa_m_hosp
    counterfactuals_dict['High_DSA_Cumulative_Hospitalization'] = dsa_m_h_hosp
    counterfactuals_dict['Low_DSA_Cumulative_Hospitalization'] = dsa_m_l_hosp

    counterfactuals_dict['Mean_DSA_Daily_Fatality'] = dsa_m_daily_fatlty
    counterfactuals_dict['High_DSA_Daily_Fatality'] = dsa_m_daily_h_fatlty
    counterfactuals_dict['Low_DSA_Daily_Fatality'] = dsa_m_daily_l_fatlty
    counterfactuals_dict['Mean_DSA_Cumulative_Fatality'] = dsa_m_fatlty
    counterfactuals_dict['High_DSA_Cumulative_Fatality'] = dsa_m_h_fatlty
    counterfactuals_dict['Low_DSA_Cumulative_Fatality'] = dsa_m_l_fatlty


    counterfactuals_dict['Diff_NoVac_Mean'] = counterfactuals_dict['Mean_Cumulative_Cases_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_NoVac_Low'] = counterfactuals_dict['Low_Cumulative_Cases_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_NoVac_High'] = counterfactuals_dict['High_Cumulative_Cases_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases']

    counterfactuals_dict['Diff_NoVac_Chld_Mean'] = counterfactuals_dict['Mean_Cumulative_Cases_Children_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_NoVac_Chld_Low'] = counterfactuals_dict['Low_Cumulative_Cases_Children_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_NoVac_Chld_High'] = counterfactuals_dict['High_Cumulative_Cases_Children_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']

    counterfactuals_dict['Diff_DSA_Mean'] = counterfactuals_dict['Mean_DSA_Cumulative'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_DSA_Low'] = counterfactuals_dict['Low_DSA_Cumulative'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_DSA_High'] = counterfactuals_dict['High_DSA_Cumulative'] - \
        counterfactuals_dict['True_Cumulative_Cases']

    counterfactuals_dict['Diff_DSA_Mean_Chld'] = counterfactuals_dict['Mean_DSA_Cumulative_Chld'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_DSA_Low_Chld'] = counterfactuals_dict['Low_DSA_Cumulative_Chld'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_DSA_High_Chld'] = counterfactuals_dict['High_DSA_Cumulative_Chld'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']

    counterfactuals_dict['Diff_Mean_Hospitalization'] = - counterfactuals_dict['True_Cumulative_Hospitalization'] + counterfactuals_dict['Mean_Cumulative_Hospitalization']
    counterfactuals_dict['Diff_Low_Hospitalization'] = - counterfactuals_dict['True_Cumulative_Hospitalization'] + counterfactuals_dict['Low_Cumulative_Hospitalization']
    counterfactuals_dict['Diff_High_Hospitalization'] = - counterfactuals_dict['True_Cumulative_Hospitalization'] + \
        counterfactuals_dict['High_Cumulative_Hospitalization']

    counterfactuals_dict['Diff_DSA_Mean_Hospitalization'] = - counterfactuals_dict['True_Cumulative_Hospitalization'] + counterfactuals_dict['Mean_DSA_Cumulative_Hospitalization']
    counterfactuals_dict['Diff_DSA_Low_Hospitalization'] = - counterfactuals_dict['True_Cumulative_Hospitalization'] + counterfactuals_dict['Low_DSA_Cumulative_Hospitalization']
    counterfactuals_dict['Diff_DSA_High_Hospitalization'] = - counterfactuals_dict['True_Cumulative_Hospitalization'] + \
        counterfactuals_dict['High_DSA_Cumulative_Hospitalization']

    counterfactuals_dict['Diff_Mean_Fatality'] = - counterfactuals_dict['True_Cumulative_Fatality'] + counterfactuals_dict['Mean_Cumulative_Fatality']
    counterfactuals_dict['Diff_Low_Fatality'] = - counterfactuals_dict['True_Cumulative_Fatality'] + counterfactuals_dict['Low_Cumulative_Fatality']
    counterfactuals_dict['Diff_High_Fatality'] = - counterfactuals_dict['True_Cumulative_Fatality'] + \
        counterfactuals_dict['High_Cumulative_Fatality']

    counterfactuals_dict['Diff_DSA_Mean_Fatality'] = - counterfactuals_dict['True_Cumulative_Fatality'] + counterfactuals_dict['Mean_DSA_Cumulative_Fatality']
    counterfactuals_dict['Diff_DSA_Low_Fatality'] = - counterfactuals_dict['True_Cumulative_Fatality'] + counterfactuals_dict['Low_DSA_Cumulative_Fatality']
    counterfactuals_dict['Diff_DSA_High_Fatality'] = - counterfactuals_dict['True_Cumulative_Fatality'] + \
        counterfactuals_dict['High_DSA_Cumulative_Fatality']

    counterfactuals_dict = pd.DataFrame(counterfactuals_dict)

    return counterfactuals_dict, fig_trajectory, fig_cum_cases, fig_prob, fig_trajectory_chld, fig_cum_cases_chld, fig_trajectory_smooth, fig_trajectory_chld_smooth, fig_hosp, fig_cum_hosp, fig_fatality, fig_cum_fatality




def vaccine_effect_whole(df, exit_df,  df_main, my_dict, dsa_parameters, confidence=0.9, drop=0.5):
    n0 = df_main.cumulative_positive.min() - df_main.daily_positive.iloc[0]
    m0 = df_main.children_cumulative_positive.min(
    ) - df_main.children_daily_positive.iloc[0]
    d0 = df_main.daily_positive.iloc[0]
    d0_Chld = df_main.children_daily_positive.iloc[0]
    upper = 1 - (1-confidence)/2
    lower = (1-confidence)/2

    trajectories = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)
    cum_cases = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)
    test_pos_probabilities = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.float64)
    chld_trajectories = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)
    chld_cum_cases = np.zeros(
        (my_dict['a'].size, df_main.time.size), dtype=np.int64)

    smooth_trajectories = np.zeros((my_dict['a'].size, df_main.time.size))
    smooth_cum_cases = np.zeros((my_dict['a'].size, df_main.time.size))
    smooth_test_pos_probabilities = np.zeros(
        (my_dict['a'].size, df_main.time.size))
    smooth_chld_trajectories = np.zeros((my_dict['a'].size, df_main.time.size))
    smooth_chld_cum_cases = np.zeros((my_dict['a'].size, df_main.time.size))

    dsa_trajectories = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_cum_trajectories = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_trajectories_chld = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)
    dsa_cum_trajectories_chld = np.zeros(
        (dsa_parameters['a'].size, df_main.time.size), dtype=np.float64)

    nDays = df_main.time.size
    time_points = np.arange(nDays)
    n = 9053900 - df_main.cumulative_positive.min()  # Israel's population
    chld_proportion = 0.360496582
    m = np.ceil(n*chld_proportion)

    for i in range(dsa_parameters['a'].size):
        a = dsa_parameters['a'].values[i]
        b = dsa_parameters['b'].values[i]
        rho = dsa_parameters['rho'].values[i]
        #epi = DSA(df=df, a=a, b=b, rho=rho)
        #n = epi.n
        def odefun(t, S): return DSA.poisson_ode_fun(t, S, a=a, b=b, rho=rho)
        t, sol = euler1d(odefun=odefun, endpoints=[0.0, nDays + 1])
        S = interp1d(t, sol)
        dsa_cum_trajectories[i] = np.asarray(
            list(n * (1 - S(x)) + n0 for x in time_points))
        dsa_trajectories[i] = np.append(d0, np.diff(dsa_cum_trajectories[i]))
        #m = np.ceil(n*chld_proportion)
        dsa_cum_trajectories_chld[i] = np.asarray(
            list(m * (1 - S(x)) + m0 for x in time_points))
        dsa_trajectories_chld[i] = np.append(
            d0_Chld, np.diff(dsa_cum_trajectories_chld[i]))

    dsa_m = np.int64(np.mean(dsa_cum_trajectories, axis=0))
    dsa_m_h = np.int64(np.quantile(dsa_cum_trajectories, q=0.975, axis=0))
    dsa_m_l = np.int64(np.quantile(dsa_cum_trajectories, q=0.025, axis=0))

    dsa_m_daily = np.int64(np.mean(dsa_trajectories, axis=0))
    dsa_m_daily_h = np.int64(np.quantile(dsa_trajectories, q=0.975, axis=0))
    dsa_m_daily_l = np.int64(np.quantile(dsa_trajectories, q=0.025, axis=0))

    dsa_m_chld = np.int64(np.mean(dsa_cum_trajectories_chld, axis=0))
    dsa_m_h_chld = np.int64(np.quantile(
        dsa_cum_trajectories_chld, q=0.975, axis=0))
    dsa_m_l_chld = np.int64(np.quantile(
        dsa_cum_trajectories_chld, q=0.025, axis=0))

    dsa_m_daily_chld = np.int64(np.mean(dsa_trajectories_chld, axis=0))
    dsa_m_daily_h_chld = np.int64(np.quantile(
        dsa_trajectories_chld, q=0.975, axis=0))
    dsa_m_daily_l_chld = np.int64(np.quantile(
        dsa_trajectories_chld, q=0.025, axis=0))

    for i in range(my_dict['a'].size):
        a = my_dict['a'].values[i]
        b = my_dict['b'].values[i]
        g = my_dict['g'].values[i]
        d = my_dict['d'].values[i]
        l = my_dict['l'].values[i]
        k = my_dict['k'].values[i]
        r_V1 = my_dict['r_V1'].values[i]
        r_V2 = my_dict['r_V2'].values[i]
        r_I = my_dict['r_I'].values[i]
        dsaobj = DSA2(df=exit_df, df_main=df_main, a=a, b=b, g=g,
                      d=0, l=l, k=k, r_V1=0.0, r_V2=0.0, r_I=r_I, drop=drop)
        trajectories[i] = dsaobj.daily_test_pos_prediction()
        cum_cases[i] = np.cumsum(trajectories[i]) + n0
        test_pos_probabilities[i] = dsaobj.daily_test_pos_probabilities()
        chld_trajectories[i] = dsaobj.children_daily_test_pos_prediction()
        chld_cum_cases[i] = np.cumsum(chld_trajectories[i]) + m0
        smooth_trajectories[i] = dsaobj.daily_test_pos_prediction_smooth()
        #smooth_trajectories[i] = pd.DataFrame(trajectories[i]).rolling(7, min_periods=1).mean().values
        smooth_cum_cases[i] = np.cumsum(smooth_trajectories[i]) + n0
        #smooth_chld_trajectories[i] = pd.DataFrame(dsaobj.children_daily_test_pos_prediction()).rolling(7, min_periods=1).mean().values
        smooth_chld_trajectories[i] = dsaobj.children_daily_test_pos_prediction_smooth()
        smooth_chld_cum_cases[i] = np.cumsum(smooth_chld_trajectories[i]) + m0

    m_trajectory = np.mean(trajectories, axis=0)
    std_trajectory = np.std(trajectories, axis=0)
    median_trajectory = np.quantile(trajectories, q=0.5, axis=0)
    high_trajectory = np.quantile(trajectories, q=upper, axis=0)
    low_trajectory = np.quantile(trajectories, q=lower, axis=0)
    # high_trajectory = m_trajectory + 1.96*std_trajectory
    # low_trajectory = m_trajectory - 1.96*std_trajectory

    m_cum_cases = np.mean(cum_cases, axis=0)
    std_cum_cases = np.std(cum_cases, axis=0)
    median_cum_cases = np.quantile(cum_cases, q=0.5, axis=0)
    high_cum_cases = np.quantile(cum_cases, q=upper, axis=0)
    low_cum_cases = np.quantile(cum_cases, q=lower, axis=0)
    # high_cum_cases = m_cum_cases + 1.96*std_cum_cases
    # low_cum_cases = m_cum_cases - 1.96*std_cum_cases

    smooth_m_trajectory = np.mean(smooth_trajectories, axis=0)
    smooth_std_trajectory = np.std(smooth_trajectories, axis=0)
    smooth_median_trajectory = np.quantile(smooth_trajectories, q=0.5, axis=0)
    smooth_high_trajectory = np.quantile(smooth_trajectories, q=upper, axis=0)
    smooth_low_trajectory = np.quantile(smooth_trajectories, q=lower, axis=0)
    # high_trajectory = m_trajectory + 1.96*std_trajectory
    # low_trajectory = m_trajectory - 1.96*std_trajectory

    smooth_m_cum_cases = np.mean(smooth_cum_cases, axis=0)
    smooth_std_cum_cases = np.std(smooth_cum_cases, axis=0)
    smooth_median_cum_cases = np.quantile(smooth_cum_cases, q=0.5, axis=0)
    smooth_high_cum_cases = np.quantile(smooth_cum_cases, q=upper, axis=0)
    smooth_low_cum_cases = np.quantile(smooth_cum_cases, q=lower, axis=0)
    # high_cum_cases = m_cum_cases + 1.96*std_cum_cases
    # low_cum_cases = m_cum_cases - 1.96*std_cum_cases

    m_prob = np.mean(test_pos_probabilities, axis=0)
    std_prob = np.std(test_pos_probabilities, axis=0)
    median_prob = np.quantile(test_pos_probabilities, q=0.5, axis=0)
    high_prob = np.quantile(test_pos_probabilities, q=upper, axis=0)
    low_prob = np.quantile(test_pos_probabilities, q=lower, axis=0)
    # high_prob = m_prob + 1.96*std_prob
    # low_prob = m_prob - 1.96*std_prob

    m_trajectory_chld = np.mean(chld_trajectories, axis=0)
    std_trajectory_chld = np.std(chld_trajectories, axis=0)
    median_trajectory_chld = np.quantile(chld_trajectories, q=0.5, axis=0)
    high_trajectory_chld = np.quantile(chld_trajectories, q=upper, axis=0)
    low_trajectory_chld = np.quantile(chld_trajectories, q=lower, axis=0)
    # high_trajectory_chld = m_trajectory_chld + 1.96*std_trajectory_chld
    # low_trajectory_chld = m_trajectory_chld - 1.96*std_trajectory_chld

    m_cum_cases_chld = np.mean(chld_cum_cases, axis=0)
    std_cum_cases_chld = np.std(chld_cum_cases, axis=0)
    median_cum_cases_chld = np.quantile(chld_cum_cases, q=0.5, axis=0)
    high_cum_cases_chld = np.quantile(chld_cum_cases, q=upper, axis=0)
    low_cum_cases_chld = np.quantile(chld_cum_cases, q=lower, axis=0)
    # high_cum_cases_chld = m_cum_cases_chld + 1.96*std_cum_cases_chld
    # low_cum_cases_chld = m_cum_cases_chld - 1.96*std_cum_cases_chld

    smooth_m_trajectory_chld = np.mean(smooth_chld_trajectories, axis=0)
    smooth_std_trajectory_chld = np.std(smooth_chld_trajectories, axis=0)
    smooth_median_trajectory_chld = np.quantile(
        smooth_chld_trajectories, q=0.5, axis=0)
    smooth_high_trajectory_chld = np.quantile(
        smooth_chld_trajectories, q=upper, axis=0)
    smooth_low_trajectory_chld = np.quantile(
        smooth_chld_trajectories, q=lower, axis=0)
    # high_trajectory_chld = m_trajectory_chld + 1.96*std_trajectory_chld
    # low_trajectory_chld = m_trajectory_chld - 1.96*std_trajectory_chld

    smooth_m_cum_cases_chld = np.mean(smooth_chld_cum_cases, axis=0)
    smooth_std_cum_cases_chld = np.std(smooth_chld_cum_cases, axis=0)
    smooth_median_cum_cases_chld = np.quantile(
        smooth_chld_cum_cases, q=0.5, axis=0)
    smooth_high_cum_cases_chld = np.quantile(
        smooth_chld_cum_cases, q=upper, axis=0)
    smooth_low_cum_cases_chld = np.quantile(
        smooth_chld_cum_cases, q=lower, axis=0)
    # high_cum_cases_chld = m_cum_cases_chld + 1.96*std_cum_cases_chld
    # low_cum_cases_chld = m_cum_cases_chld - 1.96*std_cum_cases_chld

    my_plot_configs()

    fig_trajectory = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_trajectory, high_trajectory,
                     alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l,
                     dsa_m_daily_h, alpha=.5, color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_trajectory, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    # sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # axes = plt.gca()
    # axes.set_xlim([xmin,xmax])
    ax.set_ylim([0,None])
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_trajectory_smooth = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, smooth_low_trajectory,
                     smooth_high_trajectory, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l,
                     dsa_m_daily_h, alpha=.5, color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, smooth_m_trajectory, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_cum_cases = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_cases, high_cum_cases,
                     alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_l, dsa_m_h,
                     alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_cases, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()

    fig_prob = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_prob, high_prob,
                     alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_pct_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_prob, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: no vaccination')
    plt.legend(handles=[l1, l2])
    plt.xlabel('Date')
    plt.ylabel('Probability of testing positive')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()

    fig_trajectory_chld = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_trajectory_chld,
                     high_trajectory_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l_chld,
                     dsa_m_daily_h_chld, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.children_daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_trajectory_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily_chld, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_trajectory_chld_smooth = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, smooth_low_trajectory_chld,
                     smooth_high_trajectory_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_daily_l_chld,
                     dsa_m_daily_h_chld, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.children_daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, smooth_m_trajectory_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_daily_chld, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)

    fig_cum_cases_chld = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_cases_chld,
                     high_cum_cases_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, dsa_m_l_chld,
                     dsa_m_h_chld, alpha=.5, color=greys['grey1'].get_rgb())

    l1, = plt.plot(df_main.time.values, df_main.children_cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_cases_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean: Approach 1')
    l3, = plt.plot(df_main.time.values, dsa_m_chld, '-.', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: Approach 2')

    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative test positives')
    sns.despine()
    ax = plt.gca()
    ax.set_ylim([0,None])
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()

    counterfactuals_dict = {}
    counterfactuals_dict['Date'] = df_main.time.values
    counterfactuals_dict['True_Daily_Cases'] = df_main.daily_positive.values
    counterfactuals_dict['True_Cumulative_Cases'] = df_main.cumulative_positive.values
    counterfactuals_dict['True_Daily_Pct_Positive'] = df_main.daily_pct_positive.values
    counterfactuals_dict['True_Daily_Cases_Chld'] = df_main.children_daily_positive.values
    counterfactuals_dict['True_Cumulative_Cases_Chld'] = df_main.children_cumulative_positive.values
    counterfactuals_dict['True_Daily_Pct_Positive_Chld'] = df_main.children_daily_pct_positive.values
    counterfactuals_dict['Mean_DSA_Cumulative'] = dsa_m
    counterfactuals_dict['Low_DSA_Cumulative'] = dsa_m_l
    counterfactuals_dict['High_DSA_Cumulative'] = dsa_m_h

    counterfactuals_dict['Mean_DSA_Cumulative_Chld'] = dsa_m_chld
    counterfactuals_dict['Low_DSA_Cumulative_Chld'] = dsa_m_l_chld
    counterfactuals_dict['High_DSA_Cumulative_Chld'] = dsa_m_h_chld

    counterfactuals_dict['Mean_Daily_Cases_NoVac'] = m_trajectory
    counterfactuals_dict['Median_Daily_Cases_NoVac'] = median_trajectory
    counterfactuals_dict['High_Daily_Cases_NoVac'] = high_trajectory
    counterfactuals_dict['Low_Daily_Cases_NoVac'] = low_trajectory
    counterfactuals_dict['Mean_Cumulative_Cases_NoVac'] = m_cum_cases
    counterfactuals_dict['High_Cumulative_Cases_NoVac'] = high_cum_cases
    counterfactuals_dict['Low_Cumulative_Cases_NoVac'] = low_cum_cases

    counterfactuals_dict['Mean_Cumulative_Cases_Children'] = m_cum_cases_chld
    counterfactuals_dict['Median_Cumulative_Cases_Children'] = median_cum_cases_chld
    counterfactuals_dict['High_Cumulative_Cases_Children'] = high_cum_cases_chld
    counterfactuals_dict['Low_Cumulative_Cases_Children'] = low_cum_cases_chld
    counterfactuals_dict['Mean_Daily_Children_NoVac'] = m_trajectory_chld
    counterfactuals_dict['Median_Daily_Children_NoVac'] = median_trajectory_chld
    counterfactuals_dict['High_Daily_Children_NoVac'] = high_trajectory_chld
    counterfactuals_dict['Low_Daily_Children_NoVac'] = low_trajectory_chld
    counterfactuals_dict['Mean_Cumulative_Cases_Children_NoVac'] = m_cum_cases_chld
    counterfactuals_dict['Median_Cumulative_Cases_Children_NoVac'] = median_cum_cases_chld
    counterfactuals_dict['High_Cumulative_Cases_Children_NoVac'] = high_cum_cases_chld
    counterfactuals_dict['Low_Cumulative_Cases_Children_NoVac'] = low_cum_cases_chld

    counterfactuals_dict['Diff_NoVac_Mean'] = counterfactuals_dict['Mean_Cumulative_Cases_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_NoVac_Low'] = counterfactuals_dict['Low_Cumulative_Cases_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_NoVac_High'] = counterfactuals_dict['High_Cumulative_Cases_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases']

    counterfactuals_dict['Diff_NoVac_Chld_Mean'] = counterfactuals_dict['Mean_Cumulative_Cases_Children_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_NoVac_Chld_Low'] = counterfactuals_dict['Low_Cumulative_Cases_Children_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_NoVac_Chld_High'] = counterfactuals_dict['High_Cumulative_Cases_Children_NoVac'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']

    counterfactuals_dict['Diff_DSA_Mean'] = counterfactuals_dict['Mean_DSA_Cumulative'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_DSA_Low'] = counterfactuals_dict['Low_DSA_Cumulative'] - \
        counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_DSA_High'] = counterfactuals_dict['High_DSA_Cumulative'] - \
        counterfactuals_dict['True_Cumulative_Cases']

    counterfactuals_dict['Diff_DSA_Mean_Chld'] = counterfactuals_dict['Mean_DSA_Cumulative_Chld'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_DSA_Low_Chld'] = counterfactuals_dict['Low_DSA_Cumulative_Chld'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_DSA_High_Chld'] = counterfactuals_dict['High_DSA_Cumulative_Chld'] - \
        counterfactuals_dict['True_Cumulative_Cases_Chld']

    counterfactuals_dict = pd.DataFrame(counterfactuals_dict)

    return counterfactuals_dict, fig_trajectory, fig_cum_cases, fig_prob, fig_trajectory_chld, fig_cum_cases_chld, fig_trajectory_smooth, fig_trajectory_chld_smooth




root_folder = os.getcwd()
data_folder = os.path.join(root_folder, 'data')
# plot_folder = os.path.join(root_folder, 'plots')
plot_folder = os.path.join(root_folder, 'counterfactual_plots_0320_2500')
from mycolours import * 

my_plot_configs()


fname = os.path.join(data_folder, "aggregate_data_Mar11.csv")
date_fields = ['time']
data_types = dict(daily_test=np.int64,
                  daily_positive=np.int64,
                  daily_pct_positive=np.float,
                  daily_vaccinated_dose1=np.int64,
                  daily_vaccinated_dose2=np.int64,
                  cumulative_positive=np.int64,
                  cumulative_dose1=np.int64,
                  cumulative_dose2=np.int64,
                  children_daily_test=np.int64,
                  children_daily_positive=np.int64, children_cumulative_positive=np.int64,
                  children_daily_pct_positive=np.float64)

df_full = pd.read_csv(fname, parse_dates=date_fields, dtype=data_types)
df_full


day0 = pd.to_datetime('2021-01-08')
last_date = pd.to_datetime('2021-02-28')
df_main = truncate_data_for_DSA(day0, last_date, df_full)
df_main


infection_times = list(i + rand.uniform() for i, y in
                       enumerate(df_main['daily_positive'].values) for z in
                       range(y.astype(int)))
infection_df = pd.DataFrame(infection_times, index=range(
    len(infection_times)), columns=['exit_time'])

exit_times = list(i + rand.uniform() for i, y in
                  enumerate(df_main['daily_positive'].values + df_main['daily_vaccinated_dose1'].values) for z in
                  range(y.astype(int)))
exit_df = pd.DataFrame(exit_times, index=range(
    len(exit_times)), columns=['exit_time'])


# fname = os.path.join(root_folder, 'RegularDSAPlots_Sep1Nov1/Israelposterior_samples_0318.csv')

fname = os.path.join(root_folder, 'RegularDSAPlots_Sep1Nov1/Israelposterior_samples_0321.csv')
data_types = dict(a=np.float64, b=np.float64, rho=np.float64)
dsa_parameters = pd.read_csv(fname)
dsa_parameters


fname = os.path.join(
    root_folder, 'plots/DSA_ABC_server_segment1_50_low_d/ABC_parameter_samples_1.csv')
data_types = dict(a=np.float64,
                  b=np.float64,
                  g=np.float64,
                  d=np.float64,
                  l=np.float64,
                  k=np.float64,
                  r_V1=np.float64,
                  r_V2=np.float64,
                  r_I=np.float64,
                  distances=np.float64)
posterior_samples = pd.read_csv(fname)
posterior_samples


a = np.quantile(posterior_samples.a.values, q=0.5)
b = np.quantile(posterior_samples.b.values, q=0.5)
g = np.quantile(posterior_samples.g.values, q=0.5)
d = np.quantile(posterior_samples.d.values, q=0.5)
l = np.quantile(posterior_samples.l.values, q=0.5)
k = np.quantile(posterior_samples.k.values, q=0.5)
r_V1 = np.quantile(posterior_samples.r_V1.values, q=0.5)
r_V2 = np.quantile(posterior_samples.r_V2.values, q=0.5)
r_I = np.quantile(posterior_samples.r_I.values, q=0.5)
print(a)
print(b)
print(g)
print(d)
print(l)
print(k)
print(r_V1)
print(r_V2)
print(r_I)

# df = pd.DataFrame(infection_times, index=range(len(infection_times)), columns=['infection'])
df = pd.DataFrame(exit_times, index=range(len(exit_times)), columns=['infection'])
sample = posterior_samples.sample(2500, replace=False)
# sample = posterior_samples

dsa_samples = dsa_parameters.sample(2500, replace=False)
# dsa_samples = dsa_parameters

# counterfactuals_dict, fig_trajectory, fig_cum_cases, fig_prob, fig_trajectory_chld, fig_cum_cases_chld = vaccine_effect_whole(exit_df, df_main, sample, dsa_samples, confidence=0.9, drop=0.5)


# counterfactuals_dict, fig_trajectory, fig_cum_cases, fig_prob, fig_trajectory_chld, fig_cum_cases_chld, fig_trajectory_smooth, fig_trajectory_chld_smooth = vaccine_effect_whole(
    # df, exit_df, df_main, sample, dsa_samples, confidence=0.75, drop=0.5)

counterfactuals_dict, fig_trajectory, fig_cum_cases, fig_prob, fig_trajectory_chld, fig_cum_cases_chld, fig_trajectory_smooth, fig_trajectory_chld_smooth, fig_hosp, fig_cum_hosp, fig_fatality, fig_cum_fatality = vaccine_effect(df, exit_df, df_main, sample, dsa_samples, fatality_rate=0.007605500972165701, hospitalization_rate=0.022627883391792468, confidence=0.75, drop=0.5)


fname = 'whole_trajectory_novac_DSA'
fig_save(fig_trajectory, plot_folder, fname)


fname = 'whole_trajectory_novac_DSA_smooth'
fig_save(fig_trajectory_smooth, plot_folder, fname)

fname = 'whole_cumulative_novac_DSA'
fig_save(fig_cum_cases, plot_folder, fname)

fname = 'whole_prob_novac_DSA'
fig_save(fig_prob, plot_folder, fname)

fname = 'whole_trajectory_novac_DSA_chld'
fig_save(fig_trajectory_chld, plot_folder, fname)


fname = 'whole_trajectory_novac_DSA_chld_smooth'
fig_save(fig_trajectory_chld_smooth, plot_folder, fname)

fname = 'whole_cumulative_novac_DSA_chld'
fig_save(fig_cum_cases_chld, plot_folder, fname)


fname = 'whole_hosp_novac'
fig_save(fig_hosp, plot_folder, fname)

fname = 'whole_cumulative_hosp_novac'
fig_save(fig_cum_hosp, plot_folder, fname)

fname = 'whole_fatality_novac'
fig_save(fig_fatality, plot_folder, fname)

fname = 'whole_cumulative_fatality_novac'
fig_save(fig_cum_fatality, plot_folder, fname)


fname = os.path.join(plot_folder, 'whole_noVac_counterfactual.csv')
counterfactuals_dict.to_csv(fname, index=False)
