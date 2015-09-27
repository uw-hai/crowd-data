import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import scipy.stats as ss
import numpy as np
import pandas as pd
import seaborn as sns
import os

from research_utils.util import tsplot_robust, ensure_dir
from load import *

class Data(object):
    def __init__(self, df):
        """Initialize.

        Args:
            df:     Must have 'worker', 'question', 'correct' columns.
                    Optionally, has 'time'.

        """
        self.df = df
        if 'time' in self.df.columns:
            self.time = True
            self.df = self.df.sort('time')
        else:
            self.time = False

    def make_data(self, outfname, time=False):
        cols = ['worker', 'question', 'correct']
        if time and self.time:
            cols += 'time'
        self.df[cols].to_csv(outfname, index=False)

    def make_plots(self, outdir):
        ensure_dir(outdir)
        self.plot_hist_n()
        plt.savefig(os.path.join(outdir, 'hist_n.png'))
        plt.close()
        self.plot_hist_accuracy()
        plt.savefig(os.path.join(outdir, 'hist_acc.png'))
        plt.close()
        self.plot_scatter_n_accuracy()
        plt.savefig(os.path.join(outdir, 'scatter_n_acc.png'))
        plt.close()

        if self.time:
            self.plot_rolling_mean_accuracy()
            plt.savefig(os.path.join(outdir, 'rolling.png'))
            plt.close()
            self.plot_rolling_mean_accuracy_ts()
            plt.savefig(os.path.join(outdir, 'rolling_ts.png'))
            plt.close()

    def plot_hist_n(self):
        ax = plt.gca()
        self.df.groupby('worker')['question'].count().hist()
        plt.xlabel('Number of questions answered')
        plt.ylabel('Number of workers')
        plt.title('')
        return ax

    def plot_hist_accuracy(self):
        ax = plt.gca()
        self.df.groupby('worker')['correct'].mean().hist()
        plt.xlabel('Accuracy')
        plt.ylabel('Number of workers')
        plt.title('')
        return ax

    def plot_scatter_n_accuracy(self):
        ax = plt.gca()
        acc = self.df.groupby('worker')['correct'].mean()
        n = self.df.groupby('worker')['question'].count()
        pd.concat([acc, n], axis=1).plot(
            kind='scatter', x='question', y='correct')
        plt.xlabel('Number of questions answered')
        plt.ylabel('Accuracy')
        plt.xlim((0, None))
        plt.ylim((0, 1))
        plt.title('')
        return ax

    def plot_rolling_mean_accuracy(self, window=50):
        for w, df in self.df.groupby('worker'):
            df = df.reset_index(drop=True)
            plt.plot(df.index, pd.rolling_mean(df['correct'], window=window),
                     color='blue', alpha=0.1)
        plt.ylim(0.5, 1)
        plt.xlim(0, None)
        plt.xlabel('Number of questions answered')
        plt.ylabel('Rolling accuracy ({} question window)'.format(window))
        return plt.gca()

    def plot_rolling_mean_accuracy_ts(self, window=50):
        ts = []
        for w, df in self.df.groupby('worker'):
            df = df.reset_index(drop=True)
            df = pd.rolling_mean(df['correct'], window=window)
            df.name = 'rolling_accuracy'
            df = df.reset_index().rename(columns={'index': 't'})
            df['condition'] = 'all'
            df['worker'] = w
            ts.append(df)
        ax, df_stat = tsplot_robust(pd.concat(ts, axis=0), time='t',
                                    condition='condition', unit='worker',
                                    value='rolling_accuracy', ci=95)
        plt.ylim(0.5, 1)
        plt.xlim(0, None)
        plt.xlabel('Number of questions answered')
        plt.ylabel('Mean rolling accuracy ({} question window)'.format(window))


if __name__ == '__main__':
    df = load_rajpal_icml15()
    for t, df in df.groupby('worker_type'):
        data = Data(df)
        data.make_plots('rajpal-{}'.format(t))
        data.make_data('rajpal-{}.csv'.format(t))

    df = load_bragg_hcomp13(positive_only=False)
    data = Data(df)
    data.make_plots('bragg')
    data.make_data('bragg.csv')

    df = load_bragg_hcomp13(positive_only=True)
    data = Data(df)
    data.make_plots('bragg-pos')
    data.make_data('bragg-pos.csv')

    df = load_lin_aaai12(workflow='tag')
    data = Data(df)
    data.make_plots('lin-tag')
    data.make_data('lin-tag.csv')

    df = load_lin_aaai12(workflow='wiki')
    data = Data(df)
    data.make_plots('lin-wiki')
    data.make_data('lin-wiki.csv')
