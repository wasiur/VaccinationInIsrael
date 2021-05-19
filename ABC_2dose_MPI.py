
from mpi4py import MPI
import pickle
import math
import time
from scipy.spatial import distance
from tabulate import tabulate
from optparse import OptionParser
from DSA_Vaccination_library import *
from mycolours import *
my_plot_configs()

assert MPI.COMM_WORLD.Get_size() > 1
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def truncate_data_for_DSA(start_date, end_date, df_main):
    n_remove = (df_main.time.max() - end_date).days
    df1 = df_main.drop(df_main.tail(n_remove).index)
    n_remove = (start_date - df_main.time.min()).days
    res = df1.loc[n_remove:]
    return res

def draw_parms_prior(a_bound=(0.09, 0.11),
                     b_bound=(1 / 5.6, 0.75),
                     g_bound=(0.5 / 5.6, 2 / 5.6),
                     d_bound=(0.4, 1.0),
                     l_bound=(0, 1e-3),
                     k_bound=(0.9/28, 1.1/28),
                     r_V1_bound=(0.15, 0.25),
                     r_V2_bound=(0, 0.2),
                     r_I_bound=(1e-6, 5e-1),
                     nSample=1):
    a_sample = np.random.uniform(low=a_bound[0], high=a_bound[1], size=nSample)
    b_sample = np.random.uniform(low=b_bound[0], high=b_bound[1], size=nSample)
    g_sample = np.random.uniform(low=g_bound[0], high=g_bound[1], size=nSample)
    d_sample = np.random.uniform(low=d_bound[0], high=d_bound[1], size=nSample)
    l_sample = np.random.uniform(low=l_bound[0], high=l_bound[1], size=nSample)
    k_sample = np.random.uniform(low=k_bound[0], high=k_bound[1], size=nSample)
    r_V1_sample = np.random.uniform(low=r_V1_bound[0], high=r_V1_bound[1], size=nSample)
    # r_V2_sample = np.random.uniform(low=r_V2_bound[0], high=r_V2_bound[1], size=nSample)
    r_V2_sample = np.random.uniform(low=r_V2_bound[0], high=r_V1_sample, size=nSample)
    r_I_sample = np.random.uniform(low=r_I_bound[0], high=r_I_bound[1], size=nSample)
    return a_sample, b_sample, g_sample, d_sample, l_sample, k_sample, r_V1_sample, r_V2_sample, r_I_sample



def summary(my_dict, ifSave=False, fname=None):
    a_samples = my_dict['a'].values
    b_samples = my_dict['b'].values
    g_samples = my_dict['g'].values
    d_samples = my_dict['d'].values
    l_samples = my_dict['l'].values
    k_samples = my_dict['k'].values
    r_V1_samples = my_dict['r_V1'].values
    r_V2_samples = my_dict['r_V2'].values
    r_I_samples = my_dict['r_I'].values
    R0_samples = my_dict['b'].values/my_dict['g'].values

    headers = ['Parameter', 'Mean', 'StdErr', '2.5%', '5%', '50%', '95%', '97.5%']
    table = [
        ['alpha (a): efficacy', np.mean(a_samples), np.std(a_samples), np.quantile(a_samples, q=0.025), np.quantile(a_samples, q=0.05),
         np.quantile(a_samples, q=0.5), np.quantile(a_samples, q=0.95), np.quantile(a_samples, q=0.975)],
        ['beta (b): infection rate', np.mean(b_samples), np.std(b_samples), np.quantile(b_samples, q=0.025),
         np.quantile(b_samples, q=0.05), np.quantile(b_samples, q=0.5), np.quantile(b_samples, q=0.95), np.quantile(b_samples, q=0.975)],
        ['gamma (g): natural recovery rate', np.mean(g_samples), np.std(g_samples), np.quantile(g_samples, q=0.025),
         np.quantile(g_samples, q=0.05), np.quantile(g_samples, q=0.5), np.quantile(g_samples, q=0.95), np.quantile(g_samples, q=0.975)],
        ['delta (d): vaccination rate', np.mean(d_samples), np.std(d_samples), np.quantile(d_samples, q=0.025),
         np.quantile(d_samples, q=0.05),
         np.quantile(d_samples, q=0.5), np.quantile(d_samples, q=0.95), np.quantile(d_samples, q=0.975)],
        ['lambda (l): vaccination rate', np.mean(l_samples), np.std(l_samples), np.quantile(l_samples, q=0.025),
         np.quantile(l_samples, q=0.05),
         np.quantile(l_samples, q=0.5), np.quantile(l_samples, q=0.95), np.quantile(l_samples, q=0.975)],
        ['kappa (k): second dose administration', np.mean(k_samples), np.std(k_samples),
         np.quantile(k_samples, q=0.025),
         np.quantile(k_samples, q=0.05),
         np.quantile(k_samples, q=0.5), np.quantile(k_samples, q=0.95), np.quantile(l_samples, q=0.975)],
        ['rho_V1: initial proportion of first dose', np.mean(r_V1_samples), np.std(r_V1_samples), np.quantile(r_V1_samples, q=0.025),
         np.quantile(r_V1_samples, q=0.05),
         np.quantile(r_V1_samples, q=0.5), np.quantile(r_V1_samples, q=0.95), np.quantile(r_V1_samples, q=0.975)],
        ['rho_V2: initial proportion of second dose', np.mean(r_V2_samples),
         np.std(r_V2_samples), np.quantile(r_V2_samples, q=0.025),
         np.quantile(r_V2_samples, q=0.05),
         np.quantile(r_V2_samples, q=0.5), np.quantile(r_V2_samples, q=0.95), np.quantile(r_V2_samples, q=0.975)],
        ['rho_I: initial proportion of infected', np.mean(r_I_samples), np.std(r_I_samples), np.quantile(r_I_samples, q=0.025),
         np.quantile(r_I_samples, q=0.05),
         np.quantile(r_I_samples, q=0.5), np.quantile(r_I_samples, q=0.95), np.quantile(r_I_samples, q=0.975)],
        ['R0', np.mean(R0_samples), np.std(R0_samples), np.quantile(R0_samples, q=0.025),
         np.quantile(R0_samples, q=0.05),
         np.quantile(R0_samples, q=0.5), np.quantile(R0_samples, q=0.95), np.quantile(R0_samples, q=0.975)]
    ]
    print(tabulate(table, headers=headers))

    if ifSave:
        str1 = '\\documentclass{article}\n \\usepackage{booktabs} \n  \\usepackage{graphicx} \n \\begin{document}\\rotatebox{90}{'
        str2 = '}\\end{document}'
        if fname == None:
            fname = 'summary.tex'
        with open(fname, 'w') as outputfile:
            outputfile.write(str1 + tabulate(table, headers=headers, tablefmt="latex_booktabs") + str2)
    return tabulate(table, headers=headers)

def posterior_histograms(my_dict):
    my_plot_configs()
    fig_a = plt.figure()
    plt.hist(my_dict['a'].values,
             bins=20, density=True,
             color=purplybrown['purplybrown4'].get_rgb()
             )
    plt.xlabel('$\\alpha$')
    plt.ylabel('Density')
    sns.despine()

    fig_b = plt.figure()
    plt.hist(my_dict['b'].values,
             bins=20, density=True,
             color=purplybrown['purplybrown4'].get_rgb()
             )
    plt.xlabel('$\\beta$')
    plt.ylabel('Density')
    sns.despine()

    fig_g = plt.figure()
    plt.hist(my_dict['g'].values,
             bins=20, density=True,
             color=purplybrown['purplybrown4'].get_rgb()
             )
    plt.xlabel('$\\gamma$')
    plt.ylabel('Density')
    sns.despine()

    fig_d = plt.figure()
    plt.hist(my_dict['d'].values,
             bins=20, density=True,
             color=purplybrown['purplybrown4'].get_rgb()
             )
    plt.xlabel('$\\delta$')
    plt.ylabel('Density')
    sns.despine()


    fig_l = plt.figure()
    plt.hist(my_dict['l'].values,
             bins=20, density=True,
             color=purplybrown['purplybrown4'].get_rgb()
             )
    plt.xlabel('$\\lambda$')
    plt.ylabel('Density')
    sns.despine()


    fig_k = plt.figure()
    plt.hist(my_dict['k'].values,
             bins=20, density=True,
             color=purplybrown['purplybrown4'].get_rgb()
             )
    plt.xlabel('$\\kappa $')
    plt.ylabel('Density')
    sns.despine()

    fig_r_I = plt.figure()
    plt.hist(my_dict['r_I'].values,
             bins=20, density=True,
             color=purplybrown['purplybrown4'].get_rgb()
             )
    plt.xlabel('$\\rho_I$')
    plt.ylabel('Density')
    sns.despine()

    fig_r_V1 = plt.figure()
    plt.hist(my_dict['r_V1'].values,
             bins=20, density=True,
             color=purplybrown['purplybrown4'].get_rgb()
             )
    plt.xlabel('$\\rho_{1}$')
    plt.ylabel('Density')
    sns.despine()


    fig_r_V2 = plt.figure()
    plt.hist(my_dict['r_V2'].values,
             bins=20, density=True,
             color=purplybrown['purplybrown4'].get_rgb()
             )
    plt.xlabel('$\\rho_{2}$')
    plt.ylabel('Density')
    sns.despine()


    fig_R0 = plt.figure()
    plt.hist(my_dict['b'].values/my_dict['g'].values,
             bins=20, density=True,
             color=purplybrown['purplybrown4'].get_rgb()
             )
    plt.xlabel('$R_0$')
    plt.ylabel('Density')
    sns.despine()

    return fig_a, fig_b, fig_g, fig_d, fig_l, fig_k, fig_r_V1, fig_r_V2, fig_r_I, fig_R0



def counterfactuals(df, df_main, my_dict, drop=0.5, confidence=0.9):
    n0 = df_main.cumulative_positive.min() - df_main.daily_positive.iloc[0]
    m0 = df_main.children_cumulative_positive.min() - df_main.children_daily_positive.iloc[0]
    upper = 1 - (1-confidence)/2
    lower = (1-confidence)/2

    trajectories = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    cum_cases = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    test_pos_probabilities = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.float64)
    # r_t = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.float64)
    trajectories_nv = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    cum_cases_nv = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    test_pos_probabilities_nv = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.float64)
    # r_t_nv = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.float64)

    trajectories_n2d = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    cum_cases_n2d = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    test_pos_probabilities_n2d = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.float64)

    chld_trajectories = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    chld_cum_cases = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    chld_trajectories_nv = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    chld_cum_cases_nv = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)

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
        dsaobj = DSA2(df=df, df_main=df_main, a=a, b=b, g=g, d=d, l=l, k=k, r_V1=r_V1, r_V2=r_V2, r_I=r_I, drop=drop)
        trajectories[i] = dsaobj.daily_test_pos_prediction()
        cum_cases[i] = np.cumsum(trajectories[i]) + n0
        test_pos_probabilities[i] = dsaobj.daily_test_pos_probabilities()
        chld_trajectories[i] = dsaobj.children_daily_test_pos_prediction()
        chld_cum_cases[i] = np.cumsum(chld_trajectories[i]) + m0


        #no vaccination scenario now
        dsaobj = DSA2(df=df, df_main=df_main, a=a, b=b, g=g, d=0.0, l=l, k=k, r_V1=0.0, r_V2=0.0, r_I=r_I, drop=drop)
        trajectories_nv[i] = dsaobj.daily_test_pos_prediction()
        cum_cases_nv[i] = np.cumsum(trajectories_nv[i]) + n0
        test_pos_probabilities_nv[i] = dsaobj.daily_test_pos_probabilities()
        chld_trajectories_nv[i] = dsaobj.children_daily_test_pos_prediction()
        chld_cum_cases_nv[i] = np.cumsum(chld_trajectories_nv[i]) + m0

        #no second dose scenario
        dsaobj = DSA2(df=df, df_main=df_main, a=a, b=b, g=g, d=d, l=l, k=0.0, r_V1=r_V1+r_V2, r_V2=0.0, r_I=r_I, drop=drop)
        trajectories_n2d[i] = dsaobj.daily_test_pos_prediction()
        cum_cases_n2d[i] = np.cumsum(trajectories_n2d[i]) + n0
        test_pos_probabilities_n2d[i] = dsaobj.daily_test_pos_probabilities()

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

    m_prob = np.mean(test_pos_probabilities, axis=0)
    std_prob = np.std(test_pos_probabilities, axis=0)
    median_prob = np.quantile(test_pos_probabilities, q=0.5, axis=0)
    high_prob = np.quantile(test_pos_probabilities, q=upper, axis=0)
    low_prob = np.quantile(test_pos_probabilities, q=lower, axis=0)
    # high_prob = m_prob + 1.96*std_prob
    # low_prob = m_prob - 1.96*std_prob


    m_trajectory_nv = np.mean(trajectories_nv, axis=0)
    std_trajectory_nv = np.std(trajectories_nv, axis=0)
    median_trajectory_nv = np.quantile(trajectories_nv, q=0.5, axis=0)
    high_trajectory_nv = np.quantile(trajectories_nv, q=upper, axis=0)
    low_trajectory_nv = np.quantile(trajectories_nv, q=lower, axis=0)
    # high_trajectory_nv = m_trajectory_nv + 1.96*std_trajectory_nv
    # low_trajectory_nv = m_trajectory_nv - 1.96*std_trajectory_nv


    m_cum_cases_nv = np.mean(cum_cases_nv, axis=0)
    std_cum_cases_nv = np.std(cum_cases_nv, axis=0)
    median_cum_cases_nv = np.quantile(cum_cases_nv, q=0.5, axis=0)
    high_cum_cases_nv = np.quantile(cum_cases_nv, q=upper, axis=0)
    low_cum_cases_nv = np.quantile(cum_cases_nv, q=lower, axis=0)
    # high_cum_cases_nv = m_cum_cases_nv + 1.96*std_cum_cases_nv
    # low_cum_cases_nv = m_cum_cases_nv - 1.96*std_cum_cases_nv


    m_prob_nv = np.mean(test_pos_probabilities_nv, axis=0)
    std_prob_nv = np.std(test_pos_probabilities_nv, axis=0)
    median_prob_nv = np.quantile(test_pos_probabilities_nv, q=0.5, axis=0)
    high_prob_nv = np.quantile(test_pos_probabilities_nv, q=upper, axis=0)
    low_prob_nv = np.quantile(test_pos_probabilities_nv, q=lower, axis=0)
    # high_prob_nv = m_prob_nv + 1.96*std_prob_nv
    # low_prob_nv = m_prob_nv - 1.96*std_prob_nv


    m_trajectory_n2d = np.mean(trajectories_n2d, axis=0)
    std_trajectory_n2d = np.std(trajectories_n2d, axis=0)
    median_trajectory_n2d = np.quantile(trajectories_n2d, q=0.5, axis=0)
    high_trajectory_n2d = np.quantile(trajectories_n2d, q=upper, axis=0)
    low_trajectory_n2d = np.quantile(trajectories_n2d, q=lower, axis=0)
    # high_trajectory_n2d = m_trajectory_n2d + 1.96*std_trajectory_n2d
    # low_trajectory_n2d = m_trajectory_n2d - 1.96*std_trajectory_n2d


    m_cum_cases_n2d = np.mean(cum_cases_n2d, axis=0)
    std_cum_cases_n2d = np.std(cum_cases_n2d, axis=0)
    median_cum_cases_n2d = np.quantile(cum_cases_n2d, q=0.5, axis=0)
    high_cum_cases_n2d = np.quantile(cum_cases_n2d, q=upper, axis=0)
    low_cum_cases_n2d = np.quantile(cum_cases_n2d, q=lower, axis=0)
    # high_cum_cases_n2d = m_cum_cases_n2d + 1.96*std_cum_cases_n2d
    # low_cum_cases_n2d = m_cum_cases_n2d - 1.96*std_cum_cases_n2d


    m_prob_n2d = np.mean(test_pos_probabilities_n2d, axis=0)
    std_prob_n2d = np.std(test_pos_probabilities_n2d, axis=0)
    median_prob_n2d = np.quantile(test_pos_probabilities_n2d, q=0.5, axis=0)
    high_prob_n2d = np.quantile(test_pos_probabilities_n2d, q=upper, axis=0)
    low_prob_n2d = np.quantile(test_pos_probabilities_n2d, q=lower, axis=0)
    # high_prob_n2d = m_prob_n2d + 1.96*std_prob_n2d
    # low_prob_n2d = m_prob_n2d - 1.96*std_prob_n2d

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

    m_trajectory_chld_nv = np.mean(chld_trajectories_nv, axis=0)
    std_trajectory_chld_nv = np.std(chld_trajectories_nv, axis=0)
    median_trajectory_chld_nv = np.quantile(chld_trajectories_nv, q=0.5, axis=0)
    high_trajectory_chld_nv = np.quantile(chld_trajectories_nv, q=upper, axis=0)
    low_trajectory_chld_nv = np.quantile(chld_trajectories_nv, q=lower, axis=0)
    # high_trajectory_chld_nv = m_trajectory_chld_nv + 1.96*std_trajectory_chld_nv
    # low_trajectory_chld_nv = m_trajectory_chld_nv - 1.96*std_trajectory_chld_nv

    m_cum_cases_chld_nv = np.mean(chld_cum_cases_nv, axis=0)
    std_cum_cases_chld_nv = np.std(chld_cum_cases_nv, axis=0)
    median_cum_cases_chld_nv = np.quantile(chld_cum_cases_nv, q=0.5, axis=0)
    high_cum_cases_chld_nv = np.quantile(chld_cum_cases_nv, q=upper, axis=0)
    low_cum_cases_chld_nv = np.quantile(chld_cum_cases_nv, q=lower, axis=0)
    # high_cum_cases_chld_nv = m_cum_cases_chld_nv + 1.96*std_cum_cases_chld_nv
    # low_cum_cases_chld_nv = m_cum_cases_chld_nv - 1.96*std_cum_cases_chld_nv



    my_plot_configs()

    fig_trajectory = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_trajectory, high_trajectory, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_trajectory_nv, high_trajectory_nv, alpha=.5,
                     color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_trajectory, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_trajectory_nv, ':', color=greys['grey4'].get_rgb(), lw=2, label='Mean - no vaccination')
    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)


    fig_cum_cases = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_cases, high_cum_cases, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_cum_cases_nv, high_cum_cases_nv, alpha=.5,
                     color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_cases, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_cum_cases_nv, ':', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: no vaccination')
    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative test positives')
    sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()


    fig_prob = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_prob, high_prob, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_prob_nv, high_prob_nv, alpha=.5,
                     color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_pct_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_prob, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_prob_nv, ':', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: no vaccination')
    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Probability of testing positive')
    sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()


    fig_trajectory_n2d = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_trajectory, high_trajectory, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_trajectory_n2d, high_trajectory_n2d, alpha=.5,
                     color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_trajectory, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_trajectory_n2d, ':', color=greys['grey4'].get_rgb(), lw=2, label='Mean - only one dose')
    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()


    fig_cum_cases_n2d = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_cases, high_cum_cases, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_cum_cases_n2d, high_cum_cases_n2d, alpha=.5,
                     color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_cases, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_cum_cases_n2d, ':', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: only one dose')
    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative test positives')
    sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()


    fig_prob_n2d = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_prob, high_prob, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_prob_n2d, high_prob_n2d, alpha=.5,
                     color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_pct_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_prob, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_prob_n2d, ':', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: only one dose')
    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Probability of testing positive')
    sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()


    fig_trajectory_combined = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_trajectory, high_trajectory, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_trajectory_n2d, high_trajectory_n2d, alpha=.5,
                     color=greys['grey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_trajectory_nv, high_trajectory_nv, alpha=.5,
                     color=coffee['coffee1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_trajectory, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_trajectory_n2d, ':', color=greys['grey4'].get_rgb(), lw=2, label='Mean: only one dose')
    l4, = plt.plot(df_main.time.values, m_trajectory_nv, '--', color=coffee['coffee4'].get_rgb(), lw=2,
                   label='Mean: no vaccination')
    plt.legend(handles=[l1, l2, l3, l4])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()


    fig_cum_cases_combined = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_cases, high_cum_cases, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_cum_cases_n2d, high_cum_cases_n2d, alpha=.5,
                     color=greys['grey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_cum_cases_nv, high_cum_cases_nv, alpha=.5,
                     color=coffee['coffee1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_cases, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_cum_cases_n2d, ':', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: only one dose')
    l4, = plt.plot(df_main.time.values, m_cum_cases_nv, '--', color=coffee['coffee4'].get_rgb(),
                   lw=2, label='Mean: no vaccination')
    plt.legend(handles=[l1, l2, l3, l4])
    plt.xlabel('Date')
    plt.ylabel('Cumulative test positives')
    sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()


    fig_prob_combined = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_prob, high_prob, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_prob_n2d, high_prob_n2d, alpha=.5,
                     color=greys['grey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_prob_nv, high_prob_nv, alpha=.5,
                     color=coffee['coffee1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.daily_pct_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_prob, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_prob_n2d, ':', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: only one dose')
    l4, = plt.plot(df_main.time.values, m_prob_nv, ':', color=coffee['coffee4'].get_rgb(),
                   lw=2, label='Mean: no vaccination')
    plt.legend(handles=[l1, l2, l3, l4])
    plt.xlabel('Date')
    plt.ylabel('Probability of testing positive')
    sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()


    fig_trajectory_chld = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_trajectory_chld, high_trajectory_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_trajectory_chld_nv, high_trajectory_chld_nv, alpha=.5,
                     color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.children_daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_trajectory_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_trajectory_chld_nv, ':', color=greys['grey4'].get_rgb(), lw=2, label='Mean - no vaccination')
    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Daily test positives')
    sns.despine()
    ax = plt.gca()
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    sns.despine()
    # fname = 'ABC_trajectory_comparison'
    # fig_save(fig_trajectory, plot_folder, fname)


    fig_cum_cases_chld = plt.figure()
    # k = np.size(trajectories, axis=0)
    # for i in range(k):
    #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
    plt.fill_between(df_main.time.values, low_cum_cases_chld, high_cum_cases_chld, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
    plt.fill_between(df_main.time.values, low_cum_cases_chld_nv, high_cum_cases_chld_nv, alpha=.5,
                     color=greys['grey1'].get_rgb())
    l1, = plt.plot(df_main.time.values, df_main.children_cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                   lw=2, label='Actual')
    l2, = plt.plot(df_main.time.values, m_cum_cases_chld, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                   lw=2, label='Mean')
    l3, = plt.plot(df_main.time.values, m_cum_cases_chld_nv, ':', color=greys['grey4'].get_rgb(),
                   lw=2, label='Mean: no vaccination')
    plt.legend(handles=[l1, l2, l3])
    plt.xlabel('Date')
    plt.ylabel('Cumulative test positives')
    sns.despine()
    ax = plt.gca()
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
    counterfactuals_dict['Mean_Daily_Cases'] = m_trajectory
    counterfactuals_dict['Median_Daily_Cases'] = median_trajectory
    counterfactuals_dict['High_Daily_Cases'] = high_trajectory
    counterfactuals_dict['Low_Daily_Cases'] = low_trajectory
    counterfactuals_dict['Mean_Cumulative_Cases'] = m_cum_cases
    counterfactuals_dict['High_Cumulative_Cases'] = high_cum_cases
    counterfactuals_dict['Low_Cumulative_Cases'] = low_cum_cases
    counterfactuals_dict['Mean_Prob'] = m_prob
    counterfactuals_dict['High_Prob'] = high_prob
    counterfactuals_dict['Low_Prob'] = low_prob
    counterfactuals_dict['Mean_Daily_Cases_NoVac'] = m_trajectory_nv
    counterfactuals_dict['Median_Daily_Cases_NoVac'] = median_trajectory_nv
    counterfactuals_dict['High_Daily_Cases_NoVac'] = high_trajectory_nv
    counterfactuals_dict['Low_Daily_Cases_NoVac'] = low_trajectory_nv
    counterfactuals_dict['Mean_Cumulative_Cases_NoVac'] = m_cum_cases_nv
    counterfactuals_dict['High_Cumulative_Cases_NoVac'] = high_cum_cases_nv
    counterfactuals_dict['Low_Cumulative_Cases_NoVac'] = low_cum_cases_nv
    counterfactuals_dict['Mean_Prob_NoVac'] = m_prob_nv
    counterfactuals_dict['High_Prob_NoVac'] = high_prob_nv
    counterfactuals_dict['Low_Prob_NoVac'] = low_prob_nv
    counterfactuals_dict['Mean_Daily_Cases_OnlyOneDose'] = m_trajectory_n2d
    counterfactuals_dict['Median_Daily_Cases_OnlyOneDose'] = median_trajectory_n2d
    counterfactuals_dict['High_Daily_Cases_OnlyOneDose'] = high_trajectory_n2d
    counterfactuals_dict['Low_Daily_Cases_OnlyOneDose'] = low_trajectory_n2d
    counterfactuals_dict['Mean_Cumulative_Cases_OnlyOneDose'] = m_cum_cases_n2d
    counterfactuals_dict['Median_Cumulative_Cases_OnlyOneDose'] = median_cum_cases_n2d
    counterfactuals_dict['High_Cumulative_Cases_OnlyOneDose'] = high_cum_cases_n2d
    counterfactuals_dict['Low_Cumulative_Cases_OnlyOneDose'] = low_cum_cases_n2d
    counterfactuals_dict['Mean_Prob_OnlyOneDose'] = m_prob_n2d
    counterfactuals_dict['Median_Prob_OnlyOneDose'] = median_prob_n2d
    counterfactuals_dict['High_Prob_OnlyOneDose'] = high_prob_n2d
    counterfactuals_dict['Low_Prob_OnlyOneDose'] = low_prob_n2d
    counterfactuals_dict['Mean_Daily_Children'] = m_trajectory_chld
    counterfactuals_dict['Median_Daily_Children'] = median_trajectory_chld
    counterfactuals_dict['High_Daily_Children'] = high_trajectory_chld
    counterfactuals_dict['Low_Daily_Children'] = low_trajectory_chld
    counterfactuals_dict['Mean_Cumulative_Cases_Children'] = m_cum_cases_chld
    counterfactuals_dict['Median_Cumulative_Cases_Children'] = median_cum_cases_chld
    counterfactuals_dict['High_Cumulative_Cases_Children'] = high_cum_cases_chld
    counterfactuals_dict['Low_Cumulative_Cases_Children'] = low_cum_cases_chld
    counterfactuals_dict['Mean_Daily_Children_NoVac'] = m_trajectory_chld_nv
    counterfactuals_dict['Median_Daily_Children_NoVac'] = median_trajectory_chld_nv
    counterfactuals_dict['High_Daily_Children_NoVac'] = high_trajectory_chld_nv
    counterfactuals_dict['Low_Daily_Children_NoVac'] = low_trajectory_chld_nv
    counterfactuals_dict['Mean_Cumulative_Cases_Children_NoVac'] = m_cum_cases_chld_nv
    counterfactuals_dict['Median_Cumulative_Cases_Children_NoVac'] = median_cum_cases_chld_nv
    counterfactuals_dict['High_Cumulative_Cases_Children_NoVac'] = high_cum_cases_chld_nv
    counterfactuals_dict['Low_Cumulative_Cases_Children_NoVac'] = low_cum_cases_chld_nv

    counterfactuals_dict['Diff_NoVac_Mean'] = counterfactuals_dict['Mean_Cumulative_Cases_NoVac'] - counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_NoVac_Low'] = counterfactuals_dict['Low_Cumulative_Cases_NoVac'] - \
                                              counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_NoVac_High'] = counterfactuals_dict['High_Cumulative_Cases_NoVac'] - \
                                              counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_OneDose_Mean'] = counterfactuals_dict['Mean_Cumulative_Cases_OnlyOneDose'] - \
                                              counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_OneDose_Low'] = counterfactuals_dict['Low_Cumulative_Cases_OnlyOneDose'] - \
                                              counterfactuals_dict['True_Cumulative_Cases']
    counterfactuals_dict['Diff_OneDose_High'] = counterfactuals_dict['High_Cumulative_Cases_OnlyOneDose'] - \
                                              counterfactuals_dict['True_Cumulative_Cases']

    counterfactuals_dict['Diff_NoVac_Chld_Mean'] = counterfactuals_dict['Mean_Cumulative_Cases_Children_NoVac'] - \
                                              counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_NoVac_Chld_Low'] = counterfactuals_dict['Low_Cumulative_Cases_Children_NoVac'] - \
                                             counterfactuals_dict['True_Cumulative_Cases_Chld']
    counterfactuals_dict['Diff_NoVac_Chld_High'] = counterfactuals_dict['High_Cumulative_Cases_Children_NoVac'] - \
                                              counterfactuals_dict['True_Cumulative_Cases_Chld']

    counterfactuals_dict = pd.DataFrame(counterfactuals_dict)

    return counterfactuals_dict, fig_trajectory, fig_cum_cases, fig_prob, fig_trajectory_n2d, fig_cum_cases_n2d, fig_prob_n2d, fig_trajectory_combined, fig_cum_cases_combined, fig_prob_combined, fig_trajectory_chld, fig_cum_cases_chld


def ABC(df, df_main,
        a_bound=(0.09, 0.11),
        b_bound=(1 / 5.6, 0.75),
        g_bound=(0.5 / 5.6, 2 / 5.6),
        d_bound=(0.4, 1.0),
        l_bound=(0, 1e-3),
        k_bound=(0.9/28,1.1/28),
        r_V1_bound=(0.15, 0.25),
        r_V2_bound=(0, 0.2),
        r_I_bound=(1e-6, 5e-1),
        nSample=100,
        accept=0.1,
        drop=0.5):

    a_sample, b_sample, g_sample, d_sample, l_sample, k_sample, r_V1_sample, r_V2_sample, r_I_sample = draw_parms_prior(a_bound=a_bound, b_bound=b_bound, g_bound=g_bound, d_bound=d_bound, l_bound=l_bound, k_bound=k_bound, r_V1_bound=r_V1_bound, r_V2_bound=r_V2_bound, r_I_bound=r_I_bound, nSample=nSample)

    distances = np.zeros(nSample, dtype=np.float64)
    for i in range(nSample):
        a = a_sample[i]
        b = b_sample[i]
        g = g_sample[i]
        d = d_sample[i]
        l = l_sample[i]
        k = k_sample[i]
        r_V1 = r_V1_sample[i]
        r_V2 = r_V2_sample[i]
        r_I = r_I_sample[i]
        # daily_positive = daily_test_pos_prediction(df_main, a, b, g, d, l, r_I, r_V)
        dsaobj = DSA2(df=df, df_main=df_main, a=a, b=b, g=g, d=d, l=l, k=k, r_V1=r_V1, r_V2=r_V2, r_I=r_I, drop=drop)
        daily_positive = dsaobj.daily_test_pos_prediction()
        chld_daily_positive = dsaobj.children_daily_test_pos_prediction()
        daily_pct_positive = dsaobj.daily_test_pos_probabilities()
        # daily_dose1_prediction =dsaobj.daily_dose1_prediction()
        # theta = [a, b, g, d, l, r_I, r_V]
        # daily_positive = DSA.daily_test_pos_prediction(df_main, theta)
        # distances[i] = distance.euclidean(daily_positive, df_main.daily_positive.values)
        # distances[i] = distance.euclidean(df_main.daily_pct_positive.values, daily_pct_positive)
        distances[i] = distance.euclidean(df_main.daily_pct_positive.rolling(3,min_periods=1).mean(), daily_pct_positive)
        # distances[i] = distance.euclidean(daily_positive, df_main.daily_positive.values) + distance.euclidean(chld_daily_positive, df_main.children_daily_positive.values)
        # + distance.euclidean(daily_dose1_prediction, df_main.daily_vaccinated_dose1.values)

    my_dict = {}
    my_dict['a'] = a_sample
    my_dict['b'] = b_sample
    my_dict['g'] = g_sample
    my_dict['d'] = d_sample
    my_dict['l'] = l_sample
    my_dict['k'] = k_sample
    my_dict['r_V1'] = r_V1_sample
    my_dict['r_V2'] = r_V2_sample
    my_dict['r_I'] = r_I_sample
    my_dict['distances'] = distances
    my_dict = pd.DataFrame(my_dict)
    cutoff = np.quantile(my_dict.distances.values, q=accept)
    idx = my_dict.distances < cutoff
    my_dict = my_dict[idx]

    n0 = df_main.cumulative_positive.min() - df_main.daily_positive.iloc[0]


    trajectories = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    cum_cases = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.int64)
    test_pos_probabilities = np.zeros((my_dict['a'].size, df_main.time.size), dtype=np.float64)
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
        dsaobj = DSA2(df=df, df_main=df_main, a=a, b=b, g=g, d=d, l=l, k=k, r_V1=r_V1, r_V2=r_V2, r_I=r_I)
        trajectories[i] = dsaobj.daily_test_pos_prediction()
        cum_cases[i] = np.cumsum(trajectories[i]) + n0
        test_pos_probabilities[i] = dsaobj.daily_test_pos_probabilities()
        # trajectories[i] = daily_test_pos_prediction(df_main, a, b, g, d, l, r_I, r_V)
        # plt.plot(trajectories[i])
    return my_dict, trajectories, cum_cases, test_pos_probabilities

def ABC_parm_only(df, df_main,
                  a_bound=(0.09, 0.11),
                  b_bound=(1 / 5.6, 0.75),
                  g_bound=(0.5 / 5.6, 2 / 5.6),
                  d_bound=(0.4, 1.0),
                  l_bound=(0, 1e-3),
                  k_bound=(0.9/28, 1.1/28),
                  r_V1_bound=(0.15, 0.25),
                  r_V2_bound=(0, 0.2),
                  r_I_bound=(1e-6, 5e-1),
                  nSample=100,
                  accept=0.1,
                  drop=0.5):

    a_sample, b_sample, g_sample, d_sample, l_sample, k_sample, r_V1_sample, r_V2_sample, r_I_sample = draw_parms_prior(a_bound=a_bound, b_bound=b_bound, g_bound=g_bound, d_bound=d_bound, l_bound=l_bound, k_bound=k_bound, r_V1_bound=r_V1_bound, r_V2_bound=r_V2_bound, r_I_bound=r_I_bound, nSample=nSample)

    distances = np.zeros(nSample, dtype=np.float64)
    for i in range(nSample):
        a = a_sample[i]
        b = b_sample[i]
        g = g_sample[i]
        d = d_sample[i]
        l = l_sample[i]
        k = k_sample[i]
        r_V1 = r_V1_sample[i]
        r_V2 = r_V2_sample[i]
        r_I = r_I_sample[i]
        # daily_positive = daily_test_pos_prediction(df_main, a, b, g, d, l, r_I, r_V)
        dsaobj = DSA2(df=df, df_main=df_main, a=a, b=b, g=g, d=d, l=l, k=k, r_V1=r_V1, r_V2=r_V2, r_I=r_I, drop=drop)
        daily_positive = dsaobj.daily_test_pos_prediction()
        # theta = [a, b, g, d, l, r_I, r_V]
        # daily_positive = DSA.daily_test_pos_prediction(df_main, theta)
        distances[i] = distance.euclidean(daily_positive, df_main.daily_positive.values)

    my_dict = {}
    my_dict['a'] = a_sample
    my_dict['b'] = b_sample
    my_dict['g'] = g_sample
    my_dict['d'] = d_sample
    my_dict['l'] = l_sample
    my_dict['k'] = k_sample
    my_dict['r_V1'] = r_V1_sample
    my_dict['r_V2'] = r_V2_sample
    my_dict['r_I'] = r_I_sample
    my_dict['distances'] = distances
    my_dict = pd.DataFrame(my_dict)
    cutoff = np.quantile(my_dict.distances.values, q=accept)
    idx = my_dict.distances < cutoff
    my_dict = my_dict[idx]

    return my_dict



def ABC_parallel(df, df_main, a_bound=(0.09, 0.11), b_bound=(1 / 5.6, 0.75), g_bound=(0.5 / 5.6, 2 / 5.6), d_bound=(0.4, 1.0),
        l_bound=(0, 1e-3), k_bound=(0.9/28, 1.1/28), r_V1_bound=(0.15, 0.25), r_V2_bound=(0,0.2),  r_I_bound=(1e-6, 5e-1), nSample=100, accept=0.1, drop=0.5):

    # my_dict = ABC_parm_only(df, df_main, a_bound=a_bound, b_bound=b_bound, g_bound=g_bound, d_bound=d_bound, l_bound=l_bound, r_V_bound=r_V_bound, r_I_bound=r_I_bound, nSample=nSample, accept=accept)

    my_dict, trajectories, cum_cases, test_pos_probabilities = ABC(df, df_main, a_bound=a_bound, b_bound=b_bound, g_bound=g_bound, d_bound=d_bound, l_bound=l_bound, k_bound=k_bound, r_V1_bound=r_V1_bound, r_V2_bound=r_V2_bound, r_I_bound=r_I_bound, nSample=nSample, accept=accept, drop=drop)

    res_dict = dict(my_dict=my_dict, trajectories=trajectories, cum_cases=cum_cases, test_pos_probabilities=test_pos_probabilities)
    comm.send(res_dict, dest=0, tag=rank)
    # comm.send(my_dict, dest=0, tag=rank)
    # comm.send(my_dict, dest=0, tag=rank)
    # comm.send(trajectories, dest=0, tag=rank)
    # comm.send(test_pos_probabilities, dest=0, tag=rank)
    print('I am done sending my stuff. My rank is ', rank)



def main():
    """
    Performs Approximate Bayesian Computation in parallel using Message Passing Interface (MPI) framework OpenMPI
    :return: Posterior samples of the parameters
    """
    usage = "Usage: mpiexec -n <threads> python ABC_MPI.py -d <datafile> -o <output_folder>"
    parser = OptionParser(usage)
    parser.add_option("-d", "--data-file", action="store", type="string", dest="datafile",
                      help="Name of the data file.")
    parser.add_option("-o", "--output-folder", action="store", dest="output_folder",
                      default="plots_dsa_bayesian", help="Name of the output folder")
    parser.add_option("-f", "--final-date", action="store", type="string", dest="last_date",
                      default=None, help="Last day of data to be used")
    parser.add_option("--segment",action="store", default=None, dest="segment", type="int",
                      help="Epidemic segment")
    parser.add_option("-p", action="store_true", default=False, dest="use_posterior",
                      help="Flag to use posterior")
    parser.add_option("--post_folder", action="store", default=None, dest="post_folder",
                      help="Name of the folder with posteior samples to inform ABC prior")
    parser.add_option("-a", "--accept", action="store", default=0.1, type="float", dest="accept",
                      help="acceptance percentage")
    parser.add_option("-N", action="store", dest="N", default=10000, type="int",
                      help="Size of the random sample")
    parser.add_option("-T", action="store", type="float", dest="T",
                      help="End of observation time", default=150.0)
    parser.add_option("--day-zero", type="string", action="store",
                      dest="day0", default=None,
                      help="Date of onset of the epidemic")
    parser.add_option("--plot", default=False, action="store_true", dest="ifPlot",
                      help="Flag if it should plot ABC trajectories")
    parser.add_option('--drop_factor', action="store", default=0.5, type="float", dest="drop")


    (options, args) = parser.parse_args()

    root_folder = os.getcwd()
    data_folder = os.path.join(root_folder, 'data')

    datafile = options.datafile
    date_fields = ['time']
    data_types = dict(daily_test=np.int64,
                      daily_positive=np.int64,
                      daily_pct_positive=np.float64,
                      daily_vaccinated_dose1=np.int64,
                      daily_vaccinated_dose2=np.int64,
                      cumulative_positive=np.int64,
                      cumulative_dose1=np.int64,
                      cumulative_dose2=np.int64,
                      children_daily_test=np.int64,
                      children_daily_positive=np.int64,
                      children_cumulative_positive=np.int64,
                      children_daily_pct_positive=np.float64)

    # data_types = dict(daily_test=np.int64,
    #                   daily_positive=np.int64,
    #                   daily_pct_positive=np.float,
    #                   daily_vaccinated_dose1=np.int64,
    #                   daily_vaccinated_dose2=np.int64,
    #                   cumulative_positive=np.int64,
    #                   cumulative_dose1=np.int64,
    #                   cumulative_dose2=np.int64)

    df_full = pd.read_csv(os.path.join(data_folder, datafile), parse_dates=date_fields, dtype=data_types)

    if rank == 0:
        print("The input data file is\n")
        print(df_full)

    # print("The data file is\n")
    # print(df_full)

    output_folder = options.output_folder

    if options.last_date is None:
        last_date = df_full.time.max()
    else:
        last_date = pd.to_datetime(options.last_date)

    if options.day0 is None:
        day0 = df_full.time.min()
    else:
        day0 = pd.to_datetime(options.day0)

    N = options.N
    T = options.T
    accept = options.accept
    ifPlot = options.ifPlot
    use_posterior = options.use_posterior
    post_folder = options.post_folder
    segment = options.segment
    drop = options.drop

    plot_folder = os.path.join(root_folder, output_folder)
    if not (os.path.exists(plot_folder)):
        os.system('mkdir %s' % plot_folder)

    today = pd.to_datetime('today')

    ## Preparing data for ABC
    df_main = truncate_data_for_DSA(day0, last_date, df_full)
    if rank == 0:
        print('Data after removing unnecessary dates\n')
        print(df_main)
    # print('Data after removing unnecessary dates\n')
    # print(df_main)

    infection_times = list(i + rand.uniform() for i, y in
                           enumerate(df_main['daily_positive'].values) for z in
                           range(y.astype(int)))
    infection_df = pd.DataFrame(infection_times, index=range(len(infection_times)), columns=['exit_time'])

    exit_times = list(i + rand.uniform() for i, y in
                      enumerate(df_main['daily_positive'].values + df_main['daily_vaccinated_dose1'].values) for z in
                      range(y.astype(int)))
    exit_df = pd.DataFrame(exit_times, index=range(len(exit_times)), columns=['exit_time'])

    if use_posterior:
        if post_folder is not None:
            folder = os.path.join(root_folder, post_folder)
        else:
            folder = plot_folder
        print('Using posterior samples to inform ABC prior\n')
        # folder = os.path.join(root_folder, post_folder)
        fname = os.path.join(folder, 'posterior_samples.csv')
        posterior_samples = pd.read_csv(fname)
        fields = ['a', 'b', 'g', 'd', 'l', 'k', 'r_I', 'r_V1', 'r_V2']
        posterior_samples = posterior_samples[fields]
        # a = posterior_samples['a'].mean()
        # b = posterior_samples['b'].mean()
        # g = posterior_samples['g'].mean()
        # d = posterior_samples['d'].mean()
        # l = posterior_samples['l'].mean()
        # k = posterior_samples['k'].mean()
        # r_I = posterior_samples['r_I'].mean()
        # r_V1 = posterior_samples['r_V1'].mean()
        # r_V2 = posterior_samples['r_V2'].mean()
        a = np.quantile(posterior_samples['a'].values, q=0.5)
        b = np.quantile(posterior_samples['b'].values, q=0.5)
        g = np.quantile(posterior_samples['g'].values, q=0.5)
        d = np.quantile(posterior_samples['d'].values, q=0.5)
        l = np.quantile(posterior_samples['l'].values, q=0.5)
        k = np.quantile(posterior_samples['k'].values, q=0.5)
        r_I = np.quantile(posterior_samples['r_I'].values, q=0.5)
        r_V1 = np.quantile(posterior_samples['r_V1'].values, q=0.5)
        r_V2 = np.quantile(posterior_samples['r_V2'].values, q=0.5)
        # print(a)
        # print(b)
        # print(g)
        # print(d)
        # print(l)
        # print(r_I)
        # print(r_V)
        # a_bound = (max(a - 1e-1, 0), a + 1e-1)
        a_bound = (0.92, 0.92)
        b_bound = (max(b - 1e-1, 0), 1.0)
        g_bound = (max(g - 1e-1, 0), g + 1e-1)
        d_bound = (min(max(d - 1e-3, 0), 0.01), 1.0)
        l_bound = (max(l-1e-1, 0), l+1e-1)
        # k_bound = (max(k-1e-1, 0), k+1e-1)
        k_bound = (1.0/21.0, 1.0/21.0)
        if segment == 1:
            r_V1_bound = (0.0, 0.30)
            r_V2_bound = (0.0, 0.0)
        else:
            r_V1_bound = (max(r_V1 - 1e-1, 0), min(r_V1 + 5e-1, 1))
            r_V2_bound = (max(r_V2 - 1e-1,0), min(r_V2 + 5e-1, 1))
        # r_V1_bound = (max(r_V1 - 3e-1, 0), min(r_V1 + 3e-1, 1))
        # r_V2_bound = (max(r_V2-2e-1,0), min(r_V2+2e-1,1))
        r_I_bound = (max(r_I - 1e-1, 0), min(r_I + 2e-1, 1))
    else:
        print('Using uniform priors for ABC\n')
        # a_bound = (0.05, 0.350)
        a_bound = (0.92, 0.92)
        b_bound = (0.05, 1.0)
        g_bound = (0.05, 0.25)
        d_bound = (0.00, 1.0)
        l_bound = (0, 0.25)
        k_bound = (1.0/21.0, 1.0/21.0)
        if segment == 1:
            r_V1_bound = (0.0, 0.3)
            r_V2_bound = (0.0, 0.0)
        else:
            r_V1_bound = (1e-3, 0.50)
            r_V2_bound = (0.0, 0.35)
        # r_V_bound = (0.05, 0.75)
        r_I_bound = (0.5e-5, 2e-1)

    size = comm.Get_size()
    if rank == 0:
        print('Size is', size)
    # print('Size is', size)

    each_process = int(N / float(size))
    start_id = rank * each_process
    end_id = (rank + 1) * each_process
    if rank == size - 1:
        end_id = N

    # comm.Barrier()

    if rank > 0:
        ABC_parallel(exit_df, df_main, a_bound=a_bound, b_bound=b_bound, g_bound=g_bound, d_bound=d_bound,
                     l_bound=l_bound, k_bound=k_bound, r_V1_bound=r_V1_bound, r_V2_bound=r_V2_bound, r_I_bound=r_I_bound, nSample=(end_id - start_id),
                     accept=accept, drop=drop)

    # ABC_parallel(exit_df, df_main, a_bound=a_bound, b_bound=b_bound, g_bound=g_bound, d_bound=d_bound, l_bound=l_bound, r_V_bound=r_V_bound, r_I_bound=r_I_bound, nSample=(end_id-start_id), accept=accept)

    # comm.Barrier()

    if rank == 0:
        print('I am rank 0')
        # my_dict = ABC_parm_only(exit_df, df_main, a_bound=a_bound, b_bound=b_bound, g_bound=g_bound, d_bound=d_bound, l_bound=l_bound, r_V_bound=r_V_bound, r_I_bound=r_I_bound, nSample=each_process, accept=accept)

        my_dict, trajectories, cum_cases,  daily_pct_positive = ABC(exit_df, df_main, a_bound=a_bound, b_bound=b_bound, g_bound=g_bound, d_bound=d_bound, l_bound=l_bound, k_bound=k_bound, r_V1_bound=r_V1_bound, r_V2_bound=r_V2_bound, r_I_bound=r_I_bound, nSample=each_process, accept=accept, drop=drop)
        #
        # colNames = ['a','b','g','d','l','r_V', 'r_I','distances']
        # my_dict = pd.DataFrame(columns=colNames)
        # trajectories = np.zeros((1, df_main.time.size), dtype=np.int64)
        # daily_pct_positive = np.zeros((1, df_main.time.size), dtype=np.float)
        for other_rank in range(1, size):
            res_temp = comm.recv(source=other_rank, tag=other_rank)
            print('Received output from rank ', other_rank)
            # res_dict = comm.recv(source=other_rank, tag=other_rank)
            res_dict = res_temp['my_dict']
            res_trajectory = res_temp['trajectories']
            res_cum_cases = res_temp['cum_cases']
            res_daily_pct_positive = res_temp['test_pos_probabilities']
            # my_dict = my_dict.append(res_temp, ignore_index=True)
            my_dict = my_dict.append(res_dict, ignore_index=True)
            trajectories = np.vstack([trajectories, res_trajectory])
            cum_cases = np.vstack([cum_cases, res_cum_cases])
            daily_pct_positive = np.vstack([daily_pct_positive, res_daily_pct_positive])

        # trajectories = np.delete(trajectories, (0), axis=0)
        # daily_pct_positive = np.delete(daily_pct_positive, (0), axis=0)

        fname = os.path.join(plot_folder, 'ABC_parameter_samples_' + str(segment)+ '.csv')
        my_dict.to_csv(fname)
        # fname = os.path.join(plot_folder, 'ABC_trajectories')
        # pd.DataFrame(trajectories).to_csv(fname)

        if ifPlot:
            m = np.mean(trajectories, axis=0)
            median = np.quantile(trajectories, q=0.5, axis=0)
            std = np.std(trajectories, axis=0)
            # high = m + 1.96 * std
            # low = m - 1.96 * std
            high = np.quantile(trajectories, q=0.95, axis=0)
            low = np.quantile(trajectories, q=0.05, axis=0)

            my_plot_configs()

            fig = plt.figure()
            k = np.size(trajectories, axis=0)
            # for i in range(k):
            #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
            plt.fill_between(df_main.time.values, low, high, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
            l1, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                           lw=2, label='Actual')
            # l1s, = plt.plot(df_main.time.values, df_main.daily_positive.values, '-', color=maroons['maroon3'].get_rgb(), lw=2, label='Actual')
            l2, = plt.plot(df_main.time.values, m, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                           lw=2, label='Mean')
            plt.legend(handles=[l1, l2])
            plt.xlabel('Date')
            plt.ylabel('Daily test positives')
            sns.despine()
            ax = plt.gca()
            date_form = DateFormatter("%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            sns.despine()
            fname = 'ABC_trajectory_comparison_' + str(segment)
            fig_save(fig, plot_folder, fname)
            plt.close(fig)

            m = np.mean(cum_cases, axis=0)
            median = np.quantile(cum_cases, q=0.5, axis=0)
            std = np.std(cum_cases, axis=0)
            # high = m + 1.96 * std
            # low = m - 1.96 * std
            high = np.quantile(cum_cases, q=0.95, axis=0)
            low = np.quantile(cum_cases, q=0.05, axis=0)

            my_plot_configs()

            fig = plt.figure()
            k = np.size(cum_cases, axis=0)
            # for i in range(k):
            #     plt.plot(df_main.time.values, trajectories[i],'-', color=pinks['pink2'].get_rgb(), lw=1)
            plt.fill_between(df_main.time.values, low, high, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
            l1, = plt.plot(df_main.time.values, df_main.cumulative_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                           lw=2, label='Actual')
            l2, = plt.plot(df_main.time.values, m, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                           lw=2, label='Mean')

            plt.legend(handles=[l1, l2])
            plt.xlabel('Date')
            plt.ylabel('Cumulative test positives')
            sns.despine()
            ax = plt.gca()
            date_form = DateFormatter("%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            sns.despine()
            fname = 'ABC_Cumulative_trajectory_comparison_' + str(segment)
            fig_save(fig, plot_folder, fname)
            plt.close(fig)


            m = np.mean(daily_pct_positive, axis=0)
            median = np.quantile(daily_pct_positive, q=0.5, axis=0)
            std = np.std(daily_pct_positive, axis=0)
            # high = m + 1.96 * std
            # low = m - 1.96 * std
            high = np.quantile(daily_pct_positive, q=0.95, axis=0)
            low = np.quantile(daily_pct_positive, q=0.05, axis=0)

            my_plot_configs()

            fig = plt.figure()
            # l1, = plt.plot(df_main.time.values, df_main.daily_pct_positive.values, '-', color=cyans['cyan5'].get_rgb(),
            #                lw=3, label='Actual')
            # k = np.size(daily_pct_positive, axis=0)
            # for i in range(k):
            #     plt.plot(df_main.time.values, daily_pct_positive[i], '-', color=pinks['pink2'].get_rgb(), lw=1)
            plt.fill_between(df_main.time.values, low, high, alpha=.5, color=bluegreys['bluegrey1'].get_rgb())
            l1, = plt.plot(df_main.time.values, df_main.daily_pct_positive.values, '-', color=maroons['maroon3'].get_rgb(),
                           lw=2, label='Actual')
            # l1s, = plt.plot(df_main.time.values, df_main.daily_pct_positive.rolling(7,min_periods=1).mean(), '-',
            #                color=maroons['maroon3'].get_rgb(), lw=2, label='Actual (smoothed)')
            l2, = plt.plot(df_main.time.values, m, '-.', color=purplybrown['purplybrown4'].get_rgb(),
                           lw=2, label='Mean')

            # plt.legend(handles=[l1, l1s, l2])
            plt.legend(handles=[l1,  l2])
            # plt.legend(handles=[l1])
            plt.xlabel('Date')
            plt.ylabel('Daily percent positive')
            ax = plt.gca()
            date_form = DateFormatter("%m-%d")
            ax.xaxis.set_major_formatter(date_form)
            sns.despine()
            fname = 'ABC_pct_pos_comparison_' + str(segment)
            fig_save(fig, plot_folder, fname)
            plt.close(fig)

            # counterfactuals_df, fig_trajectory, fig_cum_cases, fig_prob = counterfactuals(exit_df, df_main, my_dict, confidence=0.75)
            counterfactuals_df, fig_trajectory, fig_cum_cases, fig_prob, fig_trajectory_n2d, fig_cum_cases_n2d, fig_prob_n2d, fig_trajectory_combined, fig_cum_cases_combined, fig_prob_combined, fig_trajectory_chld, fig_cum_cases_chld = counterfactuals(exit_df, df_main, my_dict, confidence=0.90)
            fname = os.path.join(plot_folder, 'counterfactuals_'+str(segment)+'.csv')
            counterfactuals_df.to_csv(fname)
            fname = 'Counterfactuals_trajectory_'+ str(segment)
            fig_save(fig_trajectory, plot_folder, fname)
            plt.close(fig_trajectory)
            fname = 'Counterfactuals_cum_cases_'+ str(segment)
            fig_save(fig_cum_cases, plot_folder, fname)
            plt.close(fig_cum_cases)
            fname = 'Counterfactuals_prob_'+ str(segment)
            fig_save(fig_prob, plot_folder, fname)
            plt.close(fig_prob)
            fname = 'Counterfactuals_trajectory_onlyonedose_' + str(segment)
            fig_save(fig_trajectory_n2d, plot_folder, fname)
            plt.close(fig_trajectory_n2d)
            fname = 'Counterfactuals_cum_cases_onlyonedose_' + str(segment)
            fig_save(fig_cum_cases_n2d, plot_folder, fname)
            plt.close(fig_cum_cases_n2d)
            fname = 'Counterfactuals_prob_onlyonedose_' + str(segment)
            fig_save(fig_prob_n2d, plot_folder, fname)
            plt.close(fig_prob_n2d)
            fname = 'Counterfactuals_trajectory_combined_' + str(segment)
            fig_save(fig_trajectory_combined, plot_folder, fname)
            plt.close(fig_trajectory_combined)
            fname = 'Counterfactuals_cum_cases_combined_' + str(segment)
            fig_save(fig_cum_cases_combined, plot_folder, fname)
            plt.close(fig_cum_cases_combined)
            fname = 'Counterfactuals_prob_combined_' + str(segment)
            fig_save(fig_prob_combined, plot_folder, fname)
            plt.close(fig_prob_combined)
            fname = 'Counterfactuals_trajectory_chld_'+ str(segment)
            fig_save(fig_trajectory_chld, plot_folder, fname)
            plt.close(fig_trajectory_chld)
            fname = 'Counterfactuals_cum_cases_chld_'+ str(segment)
            fig_save(fig_cum_cases_chld, plot_folder, fname)
            plt.close(fig_cum_cases_chld)

            fig_a, fig_b, fig_g, fig_d, fig_l, fig_k, fig_r_V1, fig_r_V2, fig_r_I, fig_R0 = posterior_histograms(my_dict)
            fname = 'ABC_posterior_a_'+ str(segment)
            fig_save(fig_a, plot_folder, fname)
            plt.close(fig_a)
            fname = 'ABC_posterior_b_'+ str(segment)
            fig_save(fig_b, plot_folder, fname)
            plt.close(fig_b)
            fname = 'ABC_posterior_g_'+ str(segment)
            fig_save(fig_g, plot_folder, fname)
            plt.close(fig_g)
            fname = 'ABC_posterior_d_'+ str(segment)
            fig_save(fig_d, plot_folder, fname)
            plt.close(fig_d)
            fname = 'ABC_posterior_l_'+ str(segment)
            fig_save(fig_l, plot_folder, fname)
            plt.close(fig_l)
            fname = 'ABC_posterior_k_'+ str(segment)
            fig_save(fig_k, plot_folder, fname)
            plt.close(fig_k)
            fname = 'ABC_posterior_r_V1_'+ str(segment)
            fig_save(fig_r_V1, plot_folder, fname)
            plt.close(fig_r_V1)
            fname = 'ABC_posterior_r_V2_'+ str(segment)
            fig_save(fig_r_V2, plot_folder, fname)
            plt.close(fig_r_V2)
            fname = 'ABC_posterior_r_I_'+ str(segment)
            fig_save(fig_r_I, plot_folder, fname)
            plt.close(fig_r_I)
            fname = 'ABC_posterior_R0_'+ str(segment)
            fig_save(fig_R0, plot_folder, fname)
            plt.close(fig_R0)

            fname = os.path.join(plot_folder, 'ABC_summary.tex')
            summary_table = summary(my_dict, ifSave=True, fname=fname)


    # else:
    #     my_dict, trajectories, test_pos_probabilities = ABC(exit_df, df_main, a_bound=a_bound, b_bound=b_bound,
    #                                                         g_bound=g_bound, d_bound=d_bound, l_bound=l_bound,
    #                                                         r_V_bound=r_V_bound, r_I_bound=r_I_bound,
    #                                                         nSample=(end_id-start_id),
    #                                                         accept=accept)
    #     res_dict = dict(my_dict=my_dict, trajectories=trajectories, test_pos_probabilities=test_pos_probabilities)
    #     comm.send(res_dict, dest=0, tag=rank)

    # comm.Barrier()




if __name__ == '__main__':
    main()
















