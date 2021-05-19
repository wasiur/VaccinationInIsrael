import os
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.spatial import distance
from scipy.stats import gaussian_kde, binom
from numpy.random import RandomState
rand = RandomState()

import pickle
import pystan

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
    fig.savefig(os.path.join (Plot_Folder, fname),dpi=500)
    fig.savefig(os.path.join (Plot_Folder, fname + "." + 'pdf'), format='pdf', Transparent=True)
    # fig.savefig(os.path.join(Plot_Folder, fname + "." + 'pdf'), format='pdf')
    fig.savefig(os.path.join (Plot_Folder, fname + "." + 'svg'), format='svg')


class DSA1():
    def __init__(self, df=None, df_main=None, a=0.1, b=0.6, g=0.2, d=0.4, l=0.0, r_I=1e-6, r_V=1e-6, parent=None, **kwargs):
        self.df = df
        self.df_main = df_main
        self.a = a
        self.b = b
        self.g = g
        self.d = d
        self.l = l
        self.r_I = r_I
        self.r_V = r_V
        self.parent = parent
        self.T = np.ceil(self.df['exit_time'].max())
        if kwargs.get('timepoints') is None:
            self.timepoints = np.linspace(0.0, self.T, 10000)
        else:
            self.timepoints = kwargs.get('timepoints')

    @classmethod
    def SVIR_ODE(cls, t, y, a, b, g, d, l):
        dydt = np.zeros(4)
        dydt[0] = -b*y[0]*y[2] - d*y[0]
        dydt[1] = d*y[0] - a*l*y[1] - (1-a)*b*y[1]*y[2]
        dydt[2] = b*y[0]*y[2] + (1-a)*b*y[1]*y[2] - g*y[2]
        dydt[3] = a*l*y[1] + g*y[2]
        return dydt

    @classmethod
    def SVIR_Extended_ODE(cls, t, y, a, b, g, d, l):
        dydt = np.zeros(5)
        dydt[0] = -b*y[0]*y[2] - d*y[0]
        dydt[1] = d*y[0] - a*l*y[1] - (1 - a)*b*y[1]*y[2]
        dydt[2] = b*y[0]*y[2] + (1 - a)*b*y[1]*y[2] - g*y[2]
        dydt[3] = a*l*y[1] + g*y[2]
        dydt[4] = -b*y[0]*y[2] - (1 - a)*b*y[1]*y[2]
        # dydt[4] = -b * y[0] * y[2]
        return dydt

    @classmethod
    def draw_parms_prior(cls, a_bound=(0.09, 0.11),
                         b_bound=(1 / 5.6, 0.75),
                         g_bound=(0.5 / 5.6, 2 / 5.6),
                         d_bound=(0.4, 1.0),
                         l_bound=(0, 1e-3),
                         r_V_bound=(0.15, 0.25),
                         r_I_bound=(1e-6, 5e-1),
                         nSample=1):
        a_sample = np.random.uniform(low=a_bound[0], high=a_bound[1], size=nSample)
        b_sample = np.random.uniform(low=b_bound[0], high=b_bound[1], size=nSample)
        g_sample = np.random.uniform(low=g_bound[0], high=g_bound[1], size=nSample)
        d_sample = np.random.uniform(low=d_bound[0], high=d_bound[1], size=nSample)
        l_sample = np.random.uniform(low=l_bound[0], high=l_bound[1], size=nSample)
        r_V_sample = np.random.uniform(low=r_V_bound[0], high=r_V_bound[1], size=nSample)
        r_I_sample = np.random.uniform(low=r_I_bound[0], high=r_I_bound[1], size=nSample)
        return a_sample, b_sample, g_sample, d_sample, l_sample, r_V_sample, r_I_sample

    @property
    def R0(self):
        return 1.0 * self.b/self.g

    @property
    def kT(self):
        if self.parent is None:
            return self.df['exit_time'].shape[0]
        else:
            return self.parent.kT

    @property
    def rescale(self):
        return 1 - self.S(self.T)

    @property
    def n(self):
        return self.kT / self.rescale

    @property
    def sT(self):
        return self.n - self.kT

    @property
    def theta(self):
        return [self.a, self.b, self.g, self.d, self.l, self.r_I, self.r_V]

    @property
    def S(self):
        a, b, g, d, l, r_I, r_V = self.theta
        t_span = [0, self.T]
        t_eval = np.linspace(0.0, self.T, 100000)
        y0 = [1.0, self.r_V, self.r_I, 0.0]
        ode_fun = lambda t, y: DSA.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l)
        sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval, dense_output=True)
        S = interp1d(t_eval, sol.y[0])
        return S

    def add_fits(self, samples):
        fits = []
        l = np.size(samples, axis=0)
        for i in range(l):
            a, b, g, d, l, r_I, r_V = samples[i]
            fit = DSA1(df=self.df, a=a, b=b, g=g, d=d, l=l, r_I=r_I, r_V=r_V, parent=self)
            fits.append(fit)
        self.fits = fits
        return self

    def compute_density(self, theta):
        a, b, g, d, l, r_I, r_V = theta
        t_span = [0, self.T]
        t_eval = np.linspace(0.0, self.T, 100000)
        y0 = [1.0, self.r_V, self.r_I, 0.0]
        ode_fun = lambda t, y: DSA1.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l)
        sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval, dense_output=True)
        S = interp1d(t_eval, sol.y[0])
        I = interp1d(t_eval, sol.y[2])
        out = []
        ST = S(self.T)
        for x in self.timepoints:
            Sx = S(x)
            Ix = I(x)
            out.append((b*Sx*Ix + d*Sx)/(1-ST))
        return out

    def plot_density_fit_posterior(self, samples):
        nSamples = np.size(samples, axis=0)
        Ds = np.zeros((nSamples, len(self.timepoints)), dtype=np.float)
        for idx in range(nSamples):
            Ds[idx] = self.compute_density(samples[idx])
        Dslow = np.quantile(Ds, q=0.025, axis=0)
        Dshigh = np.quantile(Ds, q=0.975, axis=0)
        Dmean = np.mean(Ds, axis=0)
        fig = plt.figure()
        plt.plot(self.timepoints, Dmean, '-', color=forrest['forrest3'].get_rgb(), lw=3)
        plt.plot(self.timepoints, Dslow, '--', color=forrest['forrest3'].get_rgb(), lw=1)
        plt.plot(self.timepoints, Dshigh, '--', color=forrest['forrest3'].get_rgb(), lw=1)
        # plt.axvline(x=self.T, color=junglegreen['green3'].get_rgb(), linestyle='-')

        mirrored_data = (2 * self.T - self.df['exit_time'].values).tolist()
        combined_data = self.df['exit_time'].values.tolist() + mirrored_data
        dense = gaussian_kde(combined_data)
        denseval = list(dense(x) * 2 for x in self.timepoints)
        plt.plot(self.timepoints, denseval, '-', color=purplybrown['purplybrown4'].get_rgb(), lw=3)
        plt.fill_between(self.timepoints, Dslow, Dshigh, alpha=.3, color=forrest['forrest1'].get_rgb())
        plt.legend()
        plt.ylabel('$-\dot{S}_t/(1-S_T)$')
        plt.xlabel('t')
        c = cumtrapz(Dmean, self.timepoints)
        ind = np.argmax(c >= 0.001)
        plt.xlim((self.timepoints[ind], self.timepoints[-1] + 1))
        sns.despine()
        return fig

    @classmethod
    def prob_test_positive(cls, t, T, theta, lag=60):
        a, b, g, d, l, r_I, r_V = theta
        # T = self.T
        t_span = [0, T + 1]
        t_eval = np.linspace(0.0, T + 1, 100000)
        y0 = [1.0, r_V, r_I, 0.0, 1.0]
        ode_fun = lambda t, y: DSA1.SVIR_Extended_ODE(t, y, a=a, b=b, g=g, d=d, l=l)
        sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval,
                                     events=None, vectorized=False, args=None)
        S = interp1d(t_eval, sol.y[0])
        S_I = interp1d(t_eval, sol.y[4])
        if t < lag:
            test_pos_prob = (1.0 - S_I(t))
            # test_pos_prob = (1.0 - S_I(t))/(1-S(T))
        else:
            test_pos_prob = (S_I(t - lag) - S_I(t))
            # test_pos_prob = (S_I(t-21) - S_I(t))/(1-S(T))
        return test_pos_prob

    @classmethod
    def binom_likelihood(cls, df_main, theta):
        nDates = df_main.time.size
        total_tests = df_main.daily_test.values
        daily_pos = df_main.daily_positive.values
        T = (df_main.time.max() - df_main.time.min()).days + 1
        loglikelihood = 0.0
        for d in range(nDates):
            test_pos_prob = DSA1.prob_test_positive(d + 1, T, theta=theta)
            loglikelihood = loglikelihood + binom.logpmf(daily_pos[d], total_tests[d], test_pos_prob, loc=0)
        return -loglikelihood

    def children_daily_test_pos_prediction(self, sample=None):
        df_main = self.df_main
        if sample is None:
            sample = self.theta
        nDates = df_main.time.size
        total_tests = df_main.children_daily_test.values
        predicted_test_pos = np.zeros(nDates, dtype=np.int64)
        T = (df_main.time.max() - df_main.time.min()).days + 1

        for d in range(nDates):
            test_pos_prob = DSA1.prob_test_positive(d + 1, T, sample)
            # print(test_pos_prob)
            predicted_test_pos[d] = np.random.binomial(total_tests[d], test_pos_prob, size=1)
        return predicted_test_pos

    def daily_test_pos_prediction(self, sample=None):
        df_main = self.df_main
        if sample is None:
            sample = self.theta

        nDates = df_main.time.size
        # dates = df_main.time.values
        total_tests = df_main.daily_test.values
        predicted_test_pos = np.zeros(nDates, dtype=np.int64)
        T = (df_main.time.max() - df_main.time.min()).days + 1

        for d in range(nDates):
            test_pos_prob = DSA.prob_test_positive(d+1, T, sample)
            # print(test_pos_prob)
            predicted_test_pos[d] = np.random.binomial(total_tests[d], test_pos_prob, size=1)
        return predicted_test_pos

    def daily_test_pos_probabilities(self, sample=None):
        df_main = self.df_main
        if sample is None:
            sample = self.theta

        nDates = df_main.time.size
        test_pos_probabilities = np.zeros(nDates, dtype=np.float64)
        T = (df_main.time.max() - df_main.time.min()).days + 1

        for d in range(nDates):
            test_pos_probabilities[d] = DSA1.prob_test_positive(d+1, T, sample)
        return test_pos_probabilities

    def compare_test_pos_probabilities(self, samples, theta=None):
        nSamples = np.size(samples, axis=0)
        dates = self.df_main.time
        nDays = len(dates)
        test_pos_probabilities = np.zeros((nSamples, nDays), dtype=np.float64)
        if theta is None:
            theta = np.mean(samples, axis=0)
        for i in range(nSamples):
            sample = samples[i]
            test_pos_probabilities[i] = self.daily_test_pos_probabilities(sample=sample)

        m = np.mean(test_pos_probabilities, axis=0)
        median = np.quantile(test_pos_probabilities, q=0.5, axis=0)
        low = np.quantile(test_pos_probabilities, q=0.025, axis=0)
        high = np.quantile(test_pos_probabilities, q=0.975, axis=0)

        my_plot_configs()
        fig = plt.figure()
        lmedian, = plt.plot(self.df_main['time'].values, median, '-.', color=forrest['forrest5'].get_rgb(), lw=3,
                            label='Median')
        lm, = plt.plot(self.df_main['time'].values, median, '-', color=forrest['forrest3'].get_rgb(), lw=3,
                       label='Mean')
        l3, = plt.plot(self.df_main['time'].values, low, '--', color=forrest['forrest2'].get_rgb(), lw=1.5)
        l4, = plt.plot(self.df_main['time'].values, high, '--', color=forrest['forrest2'].get_rgb(), lw=1.5)
        # l5, = plt.fill_between(self.df_main['time'].values, low, high, alpha=.1, color=forrest['forrest1'].get_rgb())
        l7, = plt.plot(self.df_main['time'].values, self.df_main['daily_pct_positive'].values, '-',
                       color=maroons['maroon3'].get_rgb(),
                       lw=2, label='Actual')
        plt.xlabel('Dates')
        plt.ylabel('Daily percent positive')
        # plt.ylim(0.0, 1.0)
        plt.legend(handles=[lmedian, l7])
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        my_dict = {}
        # my_dict['Dates'] = dates['d']
        my_dict['Dates'] = dates
        my_dict['Mean'] = m
        # my_dict['MLE'] = mle
        my_dict['Median'] = median
        my_dict['High'] = high
        my_dict['Low'] = low
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig, my_dict

    def compare_fit_binomial(self, samples, theta=None):
        nSamples = np.size(samples, axis=0)
        dates = self.df_main.time
        nDays = len(dates)
        time_points = np.arange(nDays)
        daily_positive = np.zeros((nSamples, nDays), dtype=np.int64)
        if theta is None:
            theta = np.mean(samples, axis=0)
        for i in range(nSamples):
            sample = samples[i]
            daily_positive[i] = self.daily_test_pos_prediction(sample)

        m = np.int64(np.mean(daily_positive, axis=0))
        median = np.int64(np.quantile(daily_positive, q=0.5, axis=0))
        low = np.int64(np.quantile(daily_positive, q=0.025, axis=0))
        high = np.int64(np.quantile(daily_positive, q=0.975, axis=0))

        my_plot_configs()
        fig = plt.figure()
        lmedian, = plt.plot(self.df_main['time'].values, median, '-.', color=forrest['forrest5'].get_rgb(), lw=3, label='Median')
        lm, = plt.plot(self.df_main['time'].values, median, '-', color=forrest['forrest3'].get_rgb(), lw=3, label='Mean')
        l3, = plt.plot(self.df_main['time'].values, low, '--', color=forrest['forrest2'].get_rgb(), lw=1.5)
        l4, = plt.plot(self.df_main['time'].values, high, '--', color=forrest['forrest2'].get_rgb(), lw=1.5)
        # l5, = plt.fill_between(self.df_main['time'].values, low, high, alpha=.1, color=forrest['forrest1'].get_rgb())
        l7, = plt.plot(self.df_main['time'].values, self.df_main['daily_positive'].values, '-', color=maroons['maroon3'].get_rgb(),
                      lw=2, label='Actual')
        plt.xlabel('Dates')
        plt.ylabel('Daily cases')
        # plt.ylim(0, 2000000)
        plt.legend(handles=[lmedian, l7])
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        my_dict = {}
        # my_dict['Dates'] = dates['d']
        my_dict['Dates'] = dates
        my_dict['Mean'] = m
        # my_dict['MLE'] = mle
        my_dict['Median'] = median
        my_dict['High'] = high
        my_dict['Low'] = low
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig, my_dict

    def compare_I(self, samples, df, dates, n0=1, d0=0, theta=None):
        nSamples = np.size(samples, axis=0)
        nDays = len(dates)
        time_points = np.arange(nDays)
        mean = np.zeros((nSamples, nDays), dtype=np.float)
        mean_daily = np.zeros((nSamples, nDays), dtype=np.float)
        if theta is not None:
            theta = np.mean(samples, axis=0)

        t_span = [0, nDays + 1]
        t_eval = np.linspace(0.0, nDays + 1, 100000)

        my_plot_configs()
        fig_a = plt.figure()
        for i in range(nSamples):
            a, b, g, d, l, r_I, r_V = samples[i]
            epi = DSA1(df=self.df, a=a, b=b, g=g, d=d, l=l, r_I=r_I, r_V=r_V)
            n = epi.n
            y0 = [1.0, r_V, r_I, 0.0]
            ode_fun = lambda t, y: DSA1.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l)
            sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
            I = interp1d(t_eval, sol.y[2])
            mean_daily[i] = np.asarray(list(n * I(x) for x in time_points))
            mean[i] = np.cumsum(mean_daily[i]) + n0

        m = np.int64(np.ceil(np.mean(mean_daily, axis=0)))
        median = np.int64(np.percentile(mean_daily, 50.0, axis=0))
        low = np.int64(np.ceil(np.quantile(mean_daily, q=0.025, axis=0)))
        high = np.int64(np.ceil(np.quantile(mean_daily, q=0.975, axis=0)))

        # a, b, g, d, l, r_I, r_V = theta
        # epi = DSA(df=self.df, a=a, b=b, g=g, d=d, l=l, r_I=r_I, r_V=r_V)
        # n = epi.n
        # y0 = [1.0, r_I, r_I, 0.0]
        # ode_fun = lambda t, y: DSA.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l)
        # sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
        # I = interp1d(t_eval, sol.y[2])
        # mle = np.asarray(list(n * I(x) for x in time_points))

        # l2mle, = plt.plot(dates['d'].dt.date, mle, '-.', color=greys['grey2'].get_rgb(), lw=3, label='Prediction')
        l2median, = plt.plot(dates['d'].dt.date, median, '-.', color=cyans['cyan5'].get_rgb(), lw=3, label='Median')
        l2, = plt.plot(dates['d'].dt.date, m, '-', color=cyans['cyan5'].get_rgb(), lw=3, label="Mean")
        l3 = plt.plot(dates['d'].dt.date, low, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l4 = plt.plot(dates['d'].dt.date, high, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l5 = plt.fill_between(dates['d'].dt.date, low, high, alpha=.1, color=cyans['cyan1'].get_rgb())
        l7 = plt.plot(df['time'].values, df['daily_positive'].values, '-', color=maroons['maroon3'].get_rgb(),
                      lw=2, label='Actual')
        plt.xlabel('Dates')
        plt.ylabel('Daily cases')
        # plt.ylim(0, 2000000)
        # plt.legend(handles=[l2mle, l7])
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        my_dict = {}
        my_dict['Dates'] = dates['d']
        # my_dict['Dates'] = dates
        my_dict['Mean'] = m
        # my_dict['MLE'] = mle
        my_dict['Median'] = median
        my_dict['High'] = high
        my_dict['Low'] = low
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig_a, my_dict


    def compare_IV(self, samples, df, dates, n0=1, d0=0, theta=None):
        nSamples = np.size(samples, axis=0)
        nDays = len(dates)
        time_points = np.arange(nDays)
        mean = np.zeros((nSamples, nDays), dtype=np.float)
        mean_daily = np.zeros((nSamples, nDays), dtype=np.float)
        if theta is not None:
            theta = np.mean(samples, axis=0)

        t_span = [0, nDays + 1]
        t_eval = np.linspace(0.0, nDays + 1, 100000)

        my_plot_configs()
        fig_a = plt.figure()
        for i in range(nSamples):
            a, b, g, d, l, r_I, r_V = samples[i]
            epi = DSA1(df=self.df, a=a, b=b, g=g, d=d, l=l, r_I=r_I, r_V=r_V)
            n = epi.n
            y0 = [1.0, r_V, r_I, 0.0]
            ode_fun = lambda t, y: DSA1.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l)
            sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
            S = interp1d(t_eval, sol.y[0])
            I = interp1d(t_eval, sol.y[2])
            mean_daily[i] = np.asarray(list(n * I(x) for x in time_points))
            mean[i] = np.asarray(list(n * (1-S(x)) + n0 for x in time_points))

        m = np.int64(np.ceil(np.mean(mean, axis=0)))
        median = np.int64(np.percentile(mean, 50.0, axis=0))
        low = np.int64(np.ceil(np.quantile(mean, q=0.025, axis=0)))
        high = np.int64(np.ceil(np.quantile(mean, q=0.975, axis=0)))

        # a, b, g, d, l, r_I, r_V = theta
        # epi = DSA(df=self.df, a=a, b=b, g=g, d=d, l=l, r_I=r_I, r_V=r_V)
        # n = epi.n
        # y0 = [1.0, r_I, r_I, 0.0]
        # ode_fun = lambda t, y: DSA.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l)
        # sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
        # S = interp1d(t_eval, sol.y[0])
        # I = interp1d(t_eval, sol.y[2])
        # # mle = np.cumsum(np.asarray(list(n * I(x) for x in time_points))) + n0
        # mle = np.asarray(list(n * (1-S(x)) + n0 for x in time_points))

        # l2mle, = plt.plot(dates['d'].dt.date, mle, '-.', color=greys['grey2'].get_rgb(), lw=3, label='Prediction')
        l2median, = plt.plot(dates['d'].dt.date, median, '-.', color=cyans['cyan5'].get_rgb(), lw=3, label='Median')
        l2, = plt.plot(dates['d'].dt.date, m, '-', color=cyans['cyan5'].get_rgb(), lw=3, label="Mean")
        l3 = plt.plot(dates['d'].dt.date, low, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l4 = plt.plot(dates['d'].dt.date, high, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l5 = plt.fill_between(dates['d'].dt.date, low, high, alpha=.1, color=cyans['cyan1'].get_rgb())
        l7 = plt.plot(df['time'].values, df['cumulative_positive'].values + df['cumulative_dose1'].values, '-', color=maroons['maroon3'].get_rgb(),
                      lw=2, label='Actual')
        plt.xlabel('Dates')
        plt.ylabel('Cumulative transfers')
        # plt.ylim(0, 2000000)
        # plt.legend(handles=[l2mle, l7])
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        my_dict = {}
        my_dict['Dates'] = dates['d']
        my_dict['Mean'] = m
        # my_dict['MLE'] = mle
        my_dict['Median'] = median
        my_dict['High'] = high
        my_dict['Low'] = low
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig_a, my_dict


    def no_vaccination_scenario(self, samples, df, dates, n0=1, d0=0, theta=None):
        nSamples = np.size(samples, axis=0)
        nDays = len(dates)
        time_points = np.arange(nDays)
        mean = np.zeros((nSamples, nDays), dtype=np.float)
        mean_daily = np.zeros((nSamples, nDays), dtype=np.float)
        mean_no_vaccination = np.zeros((nSamples, nDays), dtype=np.float)
        mean_daily_no_vaccination = np.zeros((nSamples, nDays), dtype=np.float)
        if theta is not None:
            theta = np.mean(samples, axis=0)

        t_span = [0, nDays + 1]
        t_eval = np.linspace(0.0, nDays + 1, 100000)

        my_plot_configs()
        fig_a = plt.figure()
        for i in range(nSamples):
            a, b, g, d, l, r_I, r_V = samples[i]
            epi = DSA(df=self.df, a=a, b=b, g=g, d=d, l=l, r_I=r_I, r_V=r_V)
            n = epi.n
            y0 = [1.0, r_V, r_I, 0.0]
            ode_fun = lambda t, y: DSA.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l)
            sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
            I = interp1d(t_eval, sol.y[2])
            mean_daily[i] = np.asarray(list(n * I(x) for x in time_points))
            mean[i] = np.cumsum(mean_daily[i]) + n0

            epi = DSA1(df=self.df, a=a, b=b, g=g, d=0.0, l=l, r_I=r_I, r_V=r_V)
            n = epi.n
            y0 = [1.0, r_V, r_I, 0.0]
            ode_fun = lambda t, y: DSA1.SVIR_ODE(t, y, a=a, b=b, g=g, d=0.0, l=l)
            sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
            I = interp1d(t_eval, sol.y[2])
            mean_daily_no_vaccination[i] = np.asarray(list(n * I(x) for x in time_points))
            mean_no_vaccination[i] = np.cumsum(mean_daily[i]) + n0

        m = np.int64(np.ceil(np.mean(mean_daily, axis=0)))
        median = np.int64(np.percentile(mean_daily, 50.0, axis=0))
        low = np.int64(np.ceil(np.quantile(mean_daily, q=0.025, axis=0)))
        high = np.int64(np.ceil(np.quantile(mean_daily, q=0.975, axis=0)))

        m_nv = np.int64(np.ceil(np.mean(mean_daily_no_vaccination, axis=0)))
        median_nv = np.int64(np.percentile(mean_daily_no_vaccination, 50.0, axis=0))
        low_nv = np.int64(np.ceil(np.quantile(mean_daily_no_vaccination, q=0.025, axis=0)))
        high_nv = np.int64(np.ceil(np.quantile(mean_daily_no_vaccination, q=0.975, axis=0)))

        # l2mle, = plt.plot(dates['d'].dt.date, mle, '-.', color=greys['grey2'].get_rgb(), lw=3, label='Prediction')
        l2median, = plt.plot(dates['d'].dt.date, median, '-.', color=cyans['cyan5'].get_rgb(), lw=3, label='Median')
        l2, = plt.plot(dates['d'].dt.date, m, '-', color=cyans['cyan5'].get_rgb(), lw=3, label="Mean")
        l3 = plt.plot(dates['d'].dt.date, low, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l4 = plt.plot(dates['d'].dt.date, high, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l5 = plt.fill_between(dates['d'].dt.date, low, high, alpha=.2, color=cyans['cyan1'].get_rgb())

        # l2mle_nv, = plt.plot(dates['d'].dt.date, mle_nv, '-.', color=greys['grey2'].get_rgb(), lw=3, label='Prediction')
        l2median_nv, = plt.plot(dates['d'].dt.date, median_nv, '-.', color=coffee['coffee4'].get_rgb(), lw=3, label='Median')
        l2, = plt.plot(dates['d'].dt.date, m_nv, '-', color=coffee['coffee4'].get_rgb(), lw=3, label="Mean")
        l3 = plt.plot(dates['d'].dt.date, low_nv, '--', color=coffee['coffee2'].get_rgb(), lw=1.5)
        l4 = plt.plot(dates['d'].dt.date, high_nv, '--', color=coffee['coffee2'].get_rgb(), lw=1.5)
        l5 = plt.fill_between(dates['d'].dt.date, low_nv, high_nv, alpha=.2, color=coffee['coffee1'].get_rgb())

        # l7 = plt.plot(df['time'].values, df['daily_positive'].values + n0, '-', color=maroons['maroon3'].get_rgb(),lw=2, label='Actual')
        plt.xlabel('Dates')
        plt.ylabel('Daily cases')
        # plt.ylim(0, 2000000)
        # plt.legend(handles=[l2mle])
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        my_dict = {}
        my_dict['Dates'] = dates['d']
        my_dict['Mean'] = m
        # my_dict['Prediction'] = mle
        my_dict['Median'] = median
        my_dict['High'] = high
        my_dict['Low'] = low
        my_dict['Mean_NoVac'] = m_nv
        # my_dict['Prediction_NoVac'] = mle_nv
        my_dict['Median_NoVac'] = median_nv
        my_dict['High_NoVac'] = high_nv
        my_dict['Low_NoVac'] = low_nv
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig_a, my_dict





class DSA2():
    def __init__(self, df=None, df_main=None, a=0.1, b=0.6, g=0.2, d=0.4, l=0.0, k=1/28, r_I=1e-6, r_V1=1e-6, r_V2=0.0, drop=0.5, parent=None, **kwargs):
        self.df = df
        self.df_main = df_main
        self.a = a
        self.b = b
        self.g = g
        self.d = d
        self.l = l
        self.k = k
        self.r_I = r_I
        self.r_V1 = r_V1
        self.r_V2 = r_V2
        self.drop = drop
        self.parent = parent
        self.T = np.ceil(self.df['exit_time'].max())
        if kwargs.get('timepoints') is None:
            self.timepoints = np.linspace(0.0, self.T, 10000)
        else:
            self.timepoints = kwargs.get('timepoints')

    @classmethod
    def SVIR_ODE(cls, t, y, a, b, g, d, l, k, drop=0.5):
        dydt = np.zeros(5)
        dydt[0] = -b*y[0]*y[3] - d*y[0]
        dydt[1] = d*y[0] - k*y[1] - (1-drop*a)*b*y[1]*y[3] - drop*a*l*y[1]
        dydt[2] = k*y[1] - (1-a)*b*y[2]*y[3] - a*l*y[2]
        dydt[3] = b*y[0]*y[3] + (1-drop*a)*b*y[1]*y[3] + (1-a)*b*y[2]*y[3] - g*y[3]
        dydt[4] = drop*a*l*y[1] + a*l*y[2] + g*y[3]
        return dydt

    @classmethod
    def SVIR_Extended_ODE(cls, t, y, a, b, g, d, l, k, drop=0.5):
        dydt = np.zeros(6)
        dydt[0] = -b*y[0]*y[3] - d*y[0]
        dydt[1] = d*y[0] - k*y[1] - (1-drop*a)*b*y[1]*y[3] - drop*a*l*y[1]
        dydt[2] = k*y[1] - (1-a)*b*y[2]*y[3] - a*l*y[2]
        dydt[3] = b*y[0]*y[3] + (1-drop*a)*b*y[1]*y[3] + (1-a)*b*y[2]*y[3] - g*y[3]
        dydt[4] = drop*a*l*y[1] + a*l*y[2] + g*y[3]
        # dydt[5] = -b*y[0]*y[3] - (1-0.5*a)*b*y[1]*y[3] - (1-a)*b*y[2]*y[3]
        dydt[5] = y[3]
        return dydt

    @classmethod
    def draw_parms_prior(cls, a_bound=(0.09, 0.11),
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

    @property
    def R0(self):
        return 1.0 * self.b/self.g

    @property
    def kT(self):
        if self.parent is None:
            return self.df['exit_time'].shape[0]
        else:
            return self.parent.kT

    @property
    def rescale(self):
        return 1 - self.S(self.T)

    @property
    def n(self):
        return self.kT / self.rescale

    @property
    def sT(self):
        return self.n - self.kT

    @property
    def theta(self):
        return [self.a, self.b, self.g, self.d, self.l, self.k, self.r_I, self.r_V1, self.r_V2]

    @property
    def S(self):
        a, b, g, d, l, k, r_I, r_V1, r_V2 = self.theta
        t_span = [0, self.T]
        t_eval = np.linspace(0.0, self.T, 100000)
        y0 = [1.0, self.r_V1, self.r_V2, self.r_I, 0.0]
        ode_fun = lambda t, y: DSA2.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l, k=k, drop=self.drop)
        sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval, dense_output=True)
        S = interp1d(t_eval, sol.y[0])
        return S

    def add_fits(self, samples):
        fits = []
        l = np.size(samples, axis=0)
        for i in range(l):
            a, b, g, d, l, k, r_I, r_V1, r_V2 = samples[i]
            fit = DSA2(df=self.df, a=a, b=b, g=g, d=d, l=l, k=k, r_I=r_I, r_V1=r_V1, r_V2=r_V2, drop=self.drop, parent=self)
            fits.append(fit)
        self.fits = fits
        return self

    def compute_density(self, theta):
        a, b, g, d, l, k, r_I, r_V1, r_V2 = theta
        t_span = [0, self.T]
        t_eval = np.linspace(0.0, self.T, 100000)
        y0 = [1.0, self.r_V1, self.r_V2, self.r_I, 0.0]
        ode_fun = lambda t, y: DSA2.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l, k=k, drop=self.drop)
        sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval, dense_output=True)
        S = interp1d(t_eval, sol.y[0])
        I = interp1d(t_eval, sol.y[3])
        out = []
        ST = S(self.T)
        for x in self.timepoints:
            Sx = S(x)
            Ix = I(x)
            out.append((b*Sx*Ix + d*Sx)/(1-ST))
        return out

    def plot_density_fit_posterior(self, samples):
        nSamples = np.size(samples, axis=0)
        Ds = np.zeros((nSamples, len(self.timepoints)), dtype=np.float)
        for idx in range(nSamples):
            Ds[idx] = self.compute_density(samples[idx])
        Dslow = np.quantile(Ds, q=0.025, axis=0)
        Dshigh = np.quantile(Ds, q=0.975, axis=0)
        Dmean = np.mean(Ds, axis=0)
        fig = plt.figure()
        plt.plot(self.timepoints, Dmean, '-', color=forrest['forrest3'].get_rgb(), lw=3)
        plt.plot(self.timepoints, Dslow, '--', color=forrest['forrest3'].get_rgb(), lw=1)
        plt.plot(self.timepoints, Dshigh, '--', color=forrest['forrest3'].get_rgb(), lw=1)
        # plt.axvline(x=self.T, color=junglegreen['green3'].get_rgb(), linestyle='-')

        mirrored_data = (2 * self.T - self.df['exit_time'].values).tolist()
        combined_data = self.df['exit_time'].values.tolist() + mirrored_data
        dense = gaussian_kde(combined_data)
        denseval = list(dense(x) * 2 for x in self.timepoints)
        plt.plot(self.timepoints, denseval, '-', color=purplybrown['purplybrown4'].get_rgb(), lw=3)
        plt.fill_between(self.timepoints, Dslow, Dshigh, alpha=.3, color=forrest['forrest1'].get_rgb())
        plt.legend()
        plt.ylabel('$-\dot{S}_t/(1-S_T)$')
        plt.xlabel('t')
        c = cumtrapz(Dmean, self.timepoints)
        ind = np.argmax(c >= 0.001)
        plt.xlim((self.timepoints[ind], self.timepoints[-1] + 1))
        sns.despine()
        return fig

    @classmethod
    def prob_test_positive(cls, t, T, theta, lag=22, drop=0.5):
        a, b, g, d, l, k, r_I, r_V1, r_V2 = theta
        # T = self.T
        t_span = [0, T + 1]
        t_eval = np.linspace(0.0, T + 1, 100000)
        y0 = [1.0, r_V1, r_V2, r_I, 0.0, 0.0]
        ode_fun = lambda t, y: DSA2.SVIR_Extended_ODE(t, y, a=a, b=b, g=g, d=d, l=l, k=k, drop=drop)
        sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
        S = interp1d(t_eval, sol.y[0])
        V_1 = interp1d(t_eval, sol.y[1])
        V_2 = interp1d(t_eval, sol.y[2])
        I = interp1d(t_eval, sol.y[3])
        S_I = interp1d(t_eval, sol.y[5])
        dose_1_efficacy = drop*a
        factor = 1 - dose_1_efficacy
        if t < lag:
            prob1 = (1-np.exp(- b*S_I(t)))*np.exp(-d*t)/(1+r_V1+r_V2+r_I)
            prob2 = r_V1*(1-np.exp(-factor*b*S_I(t)))*np.exp(-(k+dose_1_efficacy*l)*t)/(1+r_V1+r_V2+r_I)
            prob3 = r_V2*(1-np.exp(-(1-a)*b*S_I(t)))*np.exp(-a*l*t)/(1+r_V1+r_V2+r_I)
            # prob4 = r_I*np.exp(-g*t)/(1+r_V1+r_V2+r_I)
            prob4 = r_I/(1+r_V1+r_V2+r_I)
            test_pos_prob = prob1 + prob2 + prob3 + prob4
            # test_pos_prob = (1.0 - S_I(t))
            # test_pos_prob = (1.0 - S_I(t))/(1-S(T))
        else:
            prob1 = (np.exp(- b*S_I(t-lag)) - np.exp(- b * S_I(t)))*np.exp(-d * t)/(1 + r_V1 + r_V2 + r_I)
            prob2 = r_V1*(np.exp(-factor*b*S_I(t-lag)) - np.exp(-factor*b*S_I(t)))*np.exp(-(k+dose_1_efficacy*l)*t)/(1+r_V1+r_V2+r_I)
            prob3 = r_V2*(np.exp(-(1-a)*b*S_I(t-lag)) - np.exp(-(1-a)*b*S_I(t))) * np.exp(-a*l*t)/(1+r_V1+r_V2+r_I)
            # prob4 = r_I * np.exp(-g * t) / (1 + r_V1 + r_V2 + r_I)
            prob4 = 0.0
            # prob4 = (b*S(t-21)+factor*b*V_1 + (1-a)*b*V_2)*I(t-21)*np.exp(-21*g)
            # pressure = S_I(t) - S_I(t-21)
            test_pos_prob = prob1 + prob2 + prob3 + prob4
            # test_pos_prob = (S_I(t - lag) - S_I(t))
            # test_pos_prob = (S_I(t-21) - S_I(t))/(1-S(T))
        # if test_pos_prob > 0:
        #     return test_pos_prob
        # else:
        #     return 0
        return test_pos_prob

    @classmethod
    def binom_likelihood(cls, df_main, theta, drop=0.5):
        nDates = df_main.time.size
        total_tests = df_main.daily_test.values
        daily_pos = df_main.daily_positive.values
        T = (df_main.time.max() - df_main.time.min()).days + 1
        loglikelihood = 0.0
        for d in range(nDates):
            test_pos_prob = DSA2.prob_test_positive(d + 1, T, theta=theta, drop=drop)
            loglikelihood = loglikelihood + binom.logpmf(daily_pos[d], total_tests[d], max(0,min(test_pos_prob,1)), loc=0)
        return -loglikelihood


    def children_daily_test_pos_prediction(self, sample=None):
        df_main = self.df_main
        if sample is None:
            sample = self.theta
        nDates = df_main.time.size
        total_tests = df_main.children_daily_test.values
        predicted_test_pos = np.zeros(nDates, dtype=np.int64)
        T = (df_main.time.max() - df_main.time.min()).days + 1

        for d in range(nDates):
            test_pos_prob = DSA2.prob_test_positive(d + 1, T, sample, drop=self.drop)
            # print(test_pos_prob)
            predicted_test_pos[d] = np.random.binomial(total_tests[d], max(0,min(test_pos_prob,1)), size=1)
        return predicted_test_pos

    def children_daily_test_pos_prediction_smooth(self, sample=None):
        df_main = self.df_main
        if sample is None:
            sample = self.theta
        nDates = df_main.time.size
        total_tests = df_main.children_daily_test.values
        predicted_test_pos = np.zeros(nDates, dtype=np.int64)
        res = np.zeros(nDates)
        T = (df_main.time.max() - df_main.time.min()).days + 1

        for d in range(nDates):
            test_pos_prob = DSA2.prob_test_positive(d + 1, T, sample, drop=self.drop)
            # print(test_pos_prob)
            predicted_test_pos[d] = np.random.binomial(total_tests[d], max(0,min(test_pos_prob,1)), size=1)
        res = pd.DataFrame(predicted_test_pos, columns=['daily_positive']).rolling(7, min_periods=1).mean().daily_positive.values
        return res 

    def daily_test_pos_prediction(self, sample=None):
        df_main = self.df_main
        if sample is None:
            sample = self.theta

        nDates = df_main.time.size
        # dates = df_main.time.values
        total_tests = df_main.daily_test.values
        predicted_test_pos = np.zeros(nDates, dtype=np.int64)
        T = (df_main.time.max() - df_main.time.min()).days + 1

        for d in range(nDates):
            test_pos_prob = DSA2.prob_test_positive(d+1, T, sample, drop=self.drop)
            # print(test_pos_prob)
            predicted_test_pos[d] = np.random.binomial(total_tests[d], max(0,min(test_pos_prob,1)), size=1)
        return predicted_test_pos

    def daily_test_pos_prediction_smooth(self, sample=None):
        df_main = self.df_main
        if sample is None:
            sample = self.theta

        nDates = df_main.time.size
        # dates = df_main.time.values
        total_tests = df_main.daily_test.values
        predicted_test_pos = np.zeros(nDates, dtype=np.int64)
        res = np.zeros(nDates)
        T = (df_main.time.max() - df_main.time.min()).days + 1

        for d in range(nDates):
            test_pos_prob = DSA2.prob_test_positive(
                d+1, T, sample, drop=self.drop)
            # print(test_pos_prob)
            predicted_test_pos[d] = np.random.binomial(
                total_tests[d], max(0, min(test_pos_prob, 1)), size=1)
        res = pd.DataFrame(predicted_test_pos, columns=['daily_positive']).rolling(
            7, min_periods=1).mean().daily_positive.values
        return res

    def daily_test_pos_probabilities(self, sample=None):
        df_main = self.df_main
        if sample is None:
            sample = self.theta

        nDates = df_main.time.size
        test_pos_probabilities = np.zeros(nDates, dtype=np.float64)
        T = (df_main.time.max() - df_main.time.min()).days + 1

        for d in range(nDates):
            test_pos_probabilities[d] = DSA2.prob_test_positive(d+1, T, sample, drop=self.drop)
        return test_pos_probabilities

    def daily_dose1_prediction(self, sample=None):
        df_main = self.df_main
        dates = self.df_main.time
        if sample is None:
            sample = self.theta

        nDays = len(dates)
        time_points = np.arange(nDays)
        T = (df_main.time.max() - df_main.time.min()).days + 1
        a, b, g, d, l, k, r_I, r_V1, r_V2 = sample
        t_span = [0, T + 1]
        t_eval = np.linspace(0.0, T + 1, 100000)
        y0 = [1.0, r_V1, r_V2, r_I, 0.0, 1.0]
        ode_fun = lambda t, y: DSA2.SVIR_Extended_ODE(t, y, a=a, b=b, g=g, d=d, l=l, k=k)
        sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
        V1 = interp1d(t_eval, sol.y[1])
        dsaobj = DSA2(df=self.df, df_main=self.df_main, a=a, b=b, g=g, d=d, l=l, k=k, r_I=r_I, r_V1=r_V1, r_V2=r_V2)
        n = dsaobj.n
        daily_dose1 = np.asarray(list(n * V1(x) for x in time_points))
        return daily_dose1


    def compare_test_pos_probabilities(self, samples, theta=None):
        nSamples = np.size(samples, axis=0)
        dates = self.df_main.time
        nDays = len(dates)
        test_pos_probabilities = np.zeros((nSamples, nDays), dtype=np.float64)
        if theta is None:
            theta = np.mean(samples, axis=0)
        for i in range(nSamples):
            sample = samples[i]
            test_pos_probabilities[i] = self.daily_test_pos_probabilities(sample=sample)

        m = np.mean(test_pos_probabilities, axis=0)
        median = np.quantile(test_pos_probabilities, q=0.5, axis=0)
        low = np.quantile(test_pos_probabilities, q=0.025, axis=0)
        high = np.quantile(test_pos_probabilities, q=0.975, axis=0)

        my_plot_configs()
        fig = plt.figure()
        lmedian, = plt.plot(self.df_main['time'].values, median, '-.', color=forrest['forrest5'].get_rgb(), lw=3,
                            label='Median')
        lm, = plt.plot(self.df_main['time'].values, median, '-', color=forrest['forrest3'].get_rgb(), lw=3,
                       label='Mean')
        l3, = plt.plot(self.df_main['time'].values, low, '--', color=forrest['forrest2'].get_rgb(), lw=1.5)
        l4, = plt.plot(self.df_main['time'].values, high, '--', color=forrest['forrest2'].get_rgb(), lw=1.5)
        # l5, = plt.fill_between(self.df_main['time'].values, low, high, alpha=.1, color=forrest['forrest1'].get_rgb())
        l7, = plt.plot(self.df_main['time'].values, self.df_main['daily_pct_positive'].values, '-',
                       color=maroons['maroon3'].get_rgb(),
                       lw=2, label='Actual')
        plt.xlabel('Dates')
        plt.ylabel('Daily percent positive')
        # plt.ylim(0.0, 1.0)
        plt.legend(handles=[lmedian, l7])
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        my_dict = {}
        # my_dict['Dates'] = dates['d']
        my_dict['Dates'] = dates
        my_dict['Mean'] = m
        # my_dict['MLE'] = mle
        my_dict['Median'] = median
        my_dict['High'] = high
        my_dict['Low'] = low
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig, my_dict

    def compare_fit_binomial(self, samples, theta=None):
        nSamples = np.size(samples, axis=0)
        dates = self.df_main.time
        nDays = len(dates)
        time_points = np.arange(nDays)
        daily_positive = np.zeros((nSamples, nDays), dtype=np.int64)
        if theta is None:
            theta = np.mean(samples, axis=0)
        for i in range(nSamples):
            sample = samples[i]
            daily_positive[i] = self.daily_test_pos_prediction(sample)

        m = np.int64(np.mean(daily_positive, axis=0))
        median = np.int64(np.quantile(daily_positive, q=0.5, axis=0))
        low = np.int64(np.quantile(daily_positive, q=0.025, axis=0))
        high = np.int64(np.quantile(daily_positive, q=0.975, axis=0))

        my_plot_configs()
        fig = plt.figure()
        lmedian, = plt.plot(self.df_main['time'].values, median, '-.', color=forrest['forrest5'].get_rgb(), lw=3, label='Median')
        lm, = plt.plot(self.df_main['time'].values, median, '-', color=forrest['forrest3'].get_rgb(), lw=3, label='Mean')
        l3, = plt.plot(self.df_main['time'].values, low, '--', color=forrest['forrest2'].get_rgb(), lw=1.5)
        l4, = plt.plot(self.df_main['time'].values, high, '--', color=forrest['forrest2'].get_rgb(), lw=1.5)
        # l5, = plt.fill_between(self.df_main['time'].values, low, high, alpha=.1, color=forrest['forrest1'].get_rgb())
        l7, = plt.plot(self.df_main['time'].values, self.df_main['daily_positive'].values, '-', color=maroons['maroon3'].get_rgb(),
                      lw=2, label='Actual')
        plt.xlabel('Dates')
        plt.ylabel('Daily cases')
        # plt.ylim(0, 2000000)
        plt.legend(handles=[lmedian, l7])
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        my_dict = {}
        # my_dict['Dates'] = dates['d']
        my_dict['Dates'] = dates
        my_dict['Mean'] = m
        # my_dict['MLE'] = mle
        my_dict['Median'] = median
        my_dict['High'] = high
        my_dict['Low'] = low
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig, my_dict

    def compare_I(self, samples, df, dates, n0=1, d0=0, theta=None):
        nSamples = np.size(samples, axis=0)
        nDays = len(dates)
        time_points = np.arange(nDays)
        mean = np.zeros((nSamples, nDays), dtype=np.float)
        mean_daily = np.zeros((nSamples, nDays), dtype=np.float)
        if theta is not None:
            theta = np.mean(samples, axis=0)

        t_span = [0, nDays + 1]
        t_eval = np.linspace(0.0, nDays + 1, 100000)

        my_plot_configs()
        fig_a = plt.figure()
        for i in range(nSamples):
            a, b, g, d, l, k, r_I, r_V1, r_V2 = samples[i]
            epi = DSA2(df=self.df, a=a, b=b, g=g, d=d, l=l, k=k, r_I=r_I, r_V1=r_V1, r_V2=r_V2)
            n = epi.n
            y0 = [1.0, r_V1, r_V2, r_I, 0.0]
            ode_fun = lambda t, y: DSA2.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l, k=k)
            sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
            I = interp1d(t_eval, sol.y[3])
            mean_daily[i] = np.asarray(list(n * I(x) for x in time_points))
            mean[i] = np.cumsum(mean_daily[i]) + n0

        m = np.int64(np.ceil(np.mean(mean_daily, axis=0)))
        median = np.int64(np.percentile(mean_daily, 50.0, axis=0))
        low = np.int64(np.ceil(np.quantile(mean_daily, q=0.025, axis=0)))
        high = np.int64(np.ceil(np.quantile(mean_daily, q=0.975, axis=0)))

        # a, b, g, d, l, r_I, r_V = theta
        # epi = DSA(df=self.df, a=a, b=b, g=g, d=d, l=l, r_I=r_I, r_V=r_V)
        # n = epi.n
        # y0 = [1.0, r_I, r_I, 0.0]
        # ode_fun = lambda t, y: DSA.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l)
        # sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
        # I = interp1d(t_eval, sol.y[2])
        # mle = np.asarray(list(n * I(x) for x in time_points))

        # l2mle, = plt.plot(dates['d'].dt.date, mle, '-.', color=greys['grey2'].get_rgb(), lw=3, label='Prediction')
        l2median, = plt.plot(dates['d'].dt.date, median, '-.', color=cyans['cyan5'].get_rgb(), lw=3, label='Median')
        l2, = plt.plot(dates['d'].dt.date, m, '-', color=cyans['cyan5'].get_rgb(), lw=3, label="Mean")
        l3 = plt.plot(dates['d'].dt.date, low, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l4 = plt.plot(dates['d'].dt.date, high, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l5 = plt.fill_between(dates['d'].dt.date, low, high, alpha=.1, color=cyans['cyan1'].get_rgb())
        l7 = plt.plot(df['time'].values, df['daily_positive'].values, '-', color=maroons['maroon3'].get_rgb(),
                      lw=2, label='Actual')
        plt.xlabel('Dates')
        plt.ylabel('Daily cases')
        # plt.ylim(0, 2000000)
        # plt.legend(handles=[l2mle, l7])
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        my_dict = {}
        my_dict['Dates'] = dates['d']
        # my_dict['Dates'] = dates
        my_dict['Mean'] = m
        # my_dict['MLE'] = mle
        my_dict['Median'] = median
        my_dict['High'] = high
        my_dict['Low'] = low
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig_a, my_dict


    def compare_IV(self, samples, df, dates, n0=1, d0=0, theta=None):
        nSamples = np.size(samples, axis=0)
        nDays = len(dates)
        time_points = np.arange(nDays)
        mean = np.zeros((nSamples, nDays), dtype=np.float)
        mean_daily = np.zeros((nSamples, nDays), dtype=np.float)
        if theta is not None:
            theta = np.mean(samples, axis=0)

        t_span = [0, nDays + 1]
        t_eval = np.linspace(0.0, nDays + 1, 100000)

        my_plot_configs()
        fig_a = plt.figure()
        for i in range(nSamples):
            a, b, g, d, l, k, r_I, r_V1, r_V2 = samples[i]
            epi = DSA2(df=self.df, a=a, b=b, g=g, d=d, l=l, k=k, r_I=r_I, r_V1=r_V1, r_V2=r_V2)
            n = epi.n
            y0 = [1.0, r_V1, r_V2, r_I, 0.0]
            ode_fun = lambda t, y: DSA2.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l, k=k)
            sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
            S = interp1d(t_eval, sol.y[0])
            I = interp1d(t_eval, sol.y[3])
            mean_daily[i] = np.asarray(list(n * I(x) for x in time_points))
            mean[i] = np.asarray(list(n * (1-S(x)) + n0 for x in time_points))

        m = np.int64(np.ceil(np.mean(mean, axis=0)))
        median = np.int64(np.percentile(mean, 50.0, axis=0))
        low = np.int64(np.ceil(np.quantile(mean, q=0.025, axis=0)))
        high = np.int64(np.ceil(np.quantile(mean, q=0.975, axis=0)))

        # a, b, g, d, l, r_I, r_V = theta
        # epi = DSA(df=self.df, a=a, b=b, g=g, d=d, l=l, r_I=r_I, r_V=r_V)
        # n = epi.n
        # y0 = [1.0, r_I, r_I, 0.0]
        # ode_fun = lambda t, y: DSA.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l)
        # sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
        # S = interp1d(t_eval, sol.y[0])
        # I = interp1d(t_eval, sol.y[2])
        # # mle = np.cumsum(np.asarray(list(n * I(x) for x in time_points))) + n0
        # mle = np.asarray(list(n * (1-S(x)) + n0 for x in time_points))

        # l2mle, = plt.plot(dates['d'].dt.date, mle, '-.', color=greys['grey2'].get_rgb(), lw=3, label='Prediction')
        l2median, = plt.plot(dates['d'].dt.date, median, '-.', color=cyans['cyan5'].get_rgb(), lw=3, label='Median')
        l2, = plt.plot(dates['d'].dt.date, m, '-', color=cyans['cyan5'].get_rgb(), lw=3, label="Mean")
        l3 = plt.plot(dates['d'].dt.date, low, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l4 = plt.plot(dates['d'].dt.date, high, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l5 = plt.fill_between(dates['d'].dt.date, low, high, alpha=.1, color=cyans['cyan1'].get_rgb())
        l7 = plt.plot(df['time'].values, df['cumulative_positive'].values + df['cumulative_dose1'].values, '-', color=maroons['maroon3'].get_rgb(),
                      lw=2, label='Actual')
        plt.xlabel('Dates')
        plt.ylabel('Cumulative transfers')
        # plt.ylim(0, 2000000)
        # plt.legend(handles=[l2mle, l7])
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        my_dict = {}
        my_dict['Dates'] = dates['d']
        my_dict['Mean'] = m
        # my_dict['MLE'] = mle
        my_dict['Median'] = median
        my_dict['High'] = high
        my_dict['Low'] = low
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig_a, my_dict


    def no_vaccination_scenario(self, samples, df, dates, n0=1, d0=0, theta=None):
        nSamples = np.size(samples, axis=0)
        nDays = len(dates)
        time_points = np.arange(nDays)
        mean = np.zeros((nSamples, nDays), dtype=np.float)
        mean_daily = np.zeros((nSamples, nDays), dtype=np.float)
        mean_no_vaccination = np.zeros((nSamples, nDays), dtype=np.float)
        mean_daily_no_vaccination = np.zeros((nSamples, nDays), dtype=np.float)
        if theta is not None:
            theta = np.mean(samples, axis=0)

        t_span = [0, nDays + 1]
        t_eval = np.linspace(0.0, nDays + 1, 100000)

        my_plot_configs()
        fig_a = plt.figure()
        for i in range(nSamples):
            a, b, g, d, l, k, r_I, r_V1, r_V2 = samples[i]
            epi = DSA2(df=self.df, a=a, b=b, g=g, d=d, l=l, k=k, r_I=r_I, r_V1=r_V1, r_V2=r_V2)
            n = epi.n
            y0 = [1.0, r_V1, r_V2, r_I, 0.0]
            ode_fun = lambda t, y: DSA2.SVIR_ODE(t, y, a=a, b=b, g=g, d=d, l=l, k=k)
            sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
            I = interp1d(t_eval, sol.y[3])
            mean_daily[i] = np.asarray(list(n * I(x) for x in time_points))
            mean[i] = np.cumsum(mean_daily[i]) + n0

            epi = DSA2(df=self.df, a=a, b=b, g=g, d=0.0, l=l, k=k, r_I=r_I, r_V1=0, r_V2=0)
            n = epi.n
            y0 = [1.0, r_V1, r_V2, r_I, 0.0]
            ode_fun = lambda t, y: DSA2.SVIR_ODE(t, y, a=a, b=b, g=g, d=0.0, l=l, k=k)
            sol = sc.integrate.solve_ivp(ode_fun, t_span, y0, method='RK45', t_eval=t_eval)
            I = interp1d(t_eval, sol.y[3])
            mean_daily_no_vaccination[i] = np.asarray(list(n * I(x) for x in time_points))
            mean_no_vaccination[i] = np.cumsum(mean_daily[i]) + n0

        m = np.int64(np.ceil(np.mean(mean_daily, axis=0)))
        median = np.int64(np.percentile(mean_daily, 50.0, axis=0))
        low = np.int64(np.ceil(np.quantile(mean_daily, q=0.025, axis=0)))
        high = np.int64(np.ceil(np.quantile(mean_daily, q=0.975, axis=0)))

        m_nv = np.int64(np.ceil(np.mean(mean_daily_no_vaccination, axis=0)))
        median_nv = np.int64(np.percentile(mean_daily_no_vaccination, 50.0, axis=0))
        low_nv = np.int64(np.ceil(np.quantile(mean_daily_no_vaccination, q=0.025, axis=0)))
        high_nv = np.int64(np.ceil(np.quantile(mean_daily_no_vaccination, q=0.975, axis=0)))

        # l2mle, = plt.plot(dates['d'].dt.date, mle, '-.', color=greys['grey2'].get_rgb(), lw=3, label='Prediction')
        l2median, = plt.plot(dates['d'].dt.date, median, '-.', color=cyans['cyan5'].get_rgb(), lw=3, label='Median')
        l2, = plt.plot(dates['d'].dt.date, m, '-', color=cyans['cyan5'].get_rgb(), lw=3, label="Mean")
        l3 = plt.plot(dates['d'].dt.date, low, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l4 = plt.plot(dates['d'].dt.date, high, '--', color=cyans['cyan3'].get_rgb(), lw=1.5)
        l5 = plt.fill_between(dates['d'].dt.date, low, high, alpha=.2, color=cyans['cyan1'].get_rgb())

        # l2mle_nv, = plt.plot(dates['d'].dt.date, mle_nv, '-.', color=greys['grey2'].get_rgb(), lw=3, label='Prediction')
        l2median_nv, = plt.plot(dates['d'].dt.date, median_nv, '-.', color=coffee['coffee4'].get_rgb(), lw=3, label='Median')
        l2, = plt.plot(dates['d'].dt.date, m_nv, '-', color=coffee['coffee4'].get_rgb(), lw=3, label="Mean")
        l3 = plt.plot(dates['d'].dt.date, low_nv, '--', color=coffee['coffee2'].get_rgb(), lw=1.5)
        l4 = plt.plot(dates['d'].dt.date, high_nv, '--', color=coffee['coffee2'].get_rgb(), lw=1.5)
        l5 = plt.fill_between(dates['d'].dt.date, low_nv, high_nv, alpha=.2, color=coffee['coffee1'].get_rgb())

        # l7 = plt.plot(df['time'].values, df['daily_positive'].values + n0, '-', color=maroons['maroon3'].get_rgb(),lw=2, label='Actual')
        plt.xlabel('Dates')
        plt.ylabel('Daily cases')
        # plt.ylim(0, 2000000)
        # plt.legend(handles=[l2mle])
        ax = plt.gca()
        date_form = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        sns.despine()
        my_dict = {}
        my_dict['Dates'] = dates['d']
        my_dict['Mean'] = m
        # my_dict['Prediction'] = mle
        my_dict['Median'] = median
        my_dict['High'] = high
        my_dict['Low'] = low
        my_dict['Mean_NoVac'] = m_nv
        # my_dict['Prediction_NoVac'] = mle_nv
        my_dict['Median_NoVac'] = median_nv
        my_dict['High_NoVac'] = high_nv
        my_dict['Low_NoVac'] = low_nv
        my_dict = pd.DataFrame(my_dict)
        # my_dict.to_csv(os.path.join(Plot_Folder, fname + '.csv'), index=False)
        return fig_a, my_dict







