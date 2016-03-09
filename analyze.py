import os
import csv
import json
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import scipy.stats as ss
import numpy as np
import pandas as pd
import seaborn as sns

from research_utils.util import tsplot_robust, ensure_dir

RAJPAL_ICML15_DIR = '/homes/gws/jbragg/data/rajpal-icml15/'
BRAGG_HCOMP13_DIR = '/homes/gws/jbragg/data/bragg-hcomp13/hcomp13-multilabel-data/'
LIN_AAAI12_DIR = '/homes/gws/jbragg/data/lin-aaai12/'
BRAGG_TEACH_DIR = '/homes/gws/jbragg/data/bragg-teach/'


class Data(object):
    """Class for retrieving and plotting HCOMP data."""

    def __init__(self, df):
        """Initialize.

        Args:
            df:     Must have 'worker', 'question', 'gt', 'answer',
                    and 'correct' columns. Optionally, has 'time'.

        """
        self.df = df
        if 'time' in self.df.columns:
            self.time = True
            self.df = self.df.sort('time')
        else:
            self.time = False
        if 'condition' not in self.df.columns:
            self.df['condition'] = 'all'

    #--------- Factory methods. -------------
    # TODO: Enable retrieval using workflow=None, as with Rajpal data.
    @classmethod
    def from_lin_aaai12(cls, data_dir=LIN_AAAI12_DIR, workflow='tag'):
        """Return dataframe with joined data for lin-aaai12.

        Args:
            workflow:   Either 'tag' or 'wiki' for different dataset.

        """
        # Load answers.
        basedir = os.path.join(data_dir, 'testingData')
        files = os.listdir(basedir)
        files = [os.path.join(basedir, f) for f in files if
                 f.startswith(workflow)]
        files_q = [int(s.split('q')[-1]) for s in files]
        dfs = []
        for f, q in zip(files, files_q):
            df = pd.read_csv(f, sep='\t', header=None,
                             names=['answer', 'worker'])
            df['question'] = q
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        # Load gold.
        df_gold = pd.read_csv(os.path.join(data_dir, 'questions'),
                              sep='\t', header=None)
        data = df_gold.iloc[::4, :].reset_index(drop=True)
        answers = df_gold.iloc[3:, :].iloc[::4, :][[0]].reset_index(drop=True)
        data.columns = ['i1', 'i2', 'sentence', 'entity1', 'entity2']
        answers.columns = ['answer']
        answers['answer'] = answers.answer.astype(int)
        df_gold = pd.concat([data, answers], axis=1)
        df_gold['question'] = df_gold.index
        df = df.join(df_gold, on='question', rsuffix='_gt')

        answer_1 = df.answer == df.entity1
        answer_2 = df.answer == df.entity2
        assert all(answer_1 != answer_2)
        df['answer'] = answer_2.astype(int)
        df['gt'] = df['answer_gt']
        df['correct'] = df['answer'] == df['gt']

        return cls(df)

    @classmethod
    def from_bragg_hcomp13(cls, data_dir=BRAGG_HCOMP13_DIR,
                           positive_only=False):
        df = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        df_gold = pd.read_csv(os.path.join(data_dir, 'gold.csv'))
        df_gold = df_gold.rename(columns={'Unnamed: 0': 'item'}).fillna(0)
        df_gold = df_gold.set_index('item')
        df_gold = df_gold.stack()
        df_gold.index = df_gold.index.set_names(['item', 'label'])
        df_gold.name = 'gt'
        df_gold = df_gold.reset_index()
        df = df.merge(df_gold, how='left', on=['item', 'label'])
        df['answer'] = df['selected'].astype(int)
        df['gt'] = df['gt'].astype(int)
        df['correct'] = df['answer'] == df['gt']

        if positive_only:
            df = df[df['gt'] == 1]

        # Construct questions out of item-label combinations.
        df['question'] = df.item + '-' + df.label
        return cls(df)

    @classmethod
    def from_bragg_teach(cls, data_dir=BRAGG_TEACH_DIR, min_questions=None, conditions=None, relations=None):
        """Return object from teaching data."""
        df = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        df = df[df.finalobservation & (df.action == 'ask')]
        df = df.rename(columns={
            'questionid': 'question',
            'workerid': 'worker',
            'observationlabel': 'answer',
            'questiongold': 'gt',
            'observationtime': 'time',
        })
        df['condition'] = df['condition'].map(json.loads)
        df['condition'] = df['condition'].map(lambda d: d['policy']['n'])
        if min_questions is not None:
            df = pd.concat([df for worker, df in df.groupby(
                'worker') if df.question.nunique() >= min_questions])
        if conditions is not None:
            df = df[df.condition.isin(conditions)]

        df['gt'] = df['gt'].map(json.loads)
        df['answer'] = df['answer'].map(json.loads)
        if relations is not None:
            for column in ['gt', 'answer']:
                df[column] = df[column].map(lambda d: dict(
                    (relation, d[relation]) for relation in relations))
        df['correct'] = df['gt'] == df['answer']

        return cls(df[['question', 'worker', 'answer', 'gt', 'time', 'correct', 'condition']])

    @classmethod
    def from_rajpal_icml15(cls, data_dir=RAJPAL_ICML15_DIR, worker_type=None):
        """Return dataframe with joined data for rajpal-icml15.

        Args:
            worker_type:    Either 'ordinary', 'normal', 'master', or
                            None (for all worker classes).

        """
        # Load gold.
        with open(os.path.join(data_dir, 'questionsEMD'), 'r') as f:
            rows = list(csv.reader(f, delimiter='\t'))
            difficulty = rows[::5]
            gt = rows[4::5]
            df_gold = pd.DataFrame([
                {'difficulty': d[0],
                 'gt': v[0]} for d, v in zip(difficulty, gt)])
            df_gold.index.names = ['question']
            df_gold = df_gold.reset_index()

        # Load answers without time stamps.
        df_acc = pd.DataFrame()
        for t in ('ordinary', 'master', 'normal'):
            df = pd.read_csv(os.path.join(
                data_dir, '{}PoolResultsCSV'.format(t.capitalize())),
                header=None)
            df = df.stack()
            df.index.set_names(['worker', 'question'], inplace=True)
            df.name = 'answer'
            df = df[df != 2].reset_index()
            df['worker_type'] = t
            df['worker'] = df['worker'].astype(int)
            if len(df_acc) > 0:
                df['worker'] += df_acc.worker.max() + 1
            df_acc = pd.concat([df_acc, df], axis=0)
        df = df_acc
        df = df.merge(df_gold, on='question', how='left')
        df['gt'] = df['gt'].astype(int)
        df['answer'] = df['answer'].astype(int)
        df['correct'] = df['gt'] == df['answer']

        if worker_type is not None:
            return cls(df[df.worker_type == worker_type])
        else:
            return cls(df)

    #--------- Plotting methods. -----------
    def make_data(self, outfname, time=False):
        cols = ['worker', 'question', 'correct']
        if time and self.time:
            cols += 'time'
        self.df[cols].to_csv(outfname, index=False)

    def make_plots(self, outdir):
        ensure_dir(outdir)
        for condition, ax in self.plot_hist_n():
            plt.savefig(os.path.join(
                outdir, 'hist_n_{}.png'.format(condition)))
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
        for condition, df in self.df.groupby('condition'):
            plt.cla()
            ax = plt.gca()
            df.groupby('worker')['question'].count().hist()
            plt.xlabel('Number of questions answered')
            plt.ylabel('Number of workers')
            plt.title('')
            yield condition, ax

    def plot_hist_accuracy(self):
        ax = plt.gca()
        self.df.groupby('worker')['correct'].mean().hist()
        plt.xlabel('Accuracy')
        plt.ylabel('Number of workers')
        plt.title('')
        return ax

    def get_mean_accuracy(self):
        """Return mean question-level accuracy."""
        return self.df.correct.mean()

    def get_n_workers(self):
        """Return number of workers."""
        return self.df.worker.nunique()

    def get_n_answers(self):
        """Return number of answers."""
        return len(self.df)

    def get_n_questions(self):
        """Return number of questions."""
        return self.df.question.nunique()

    def plot_scatter_n_accuracy(self):
        ax = plt.gca()
        acc = self.df.groupby('worker')['correct'].mean()
        n = self.df.groupby('worker')['question'].count()
        condition = self.df.groupby('worker')['condition'].first()
        df = pd.concat([acc, n, condition], axis=1)
        sns.lmplot('question', 'correct', data=df, hue='condition',
                   fit_reg=False)
        plt.xlabel('Number of questions answered')
        plt.ylabel('Accuracy')
        plt.xlim((0, None))
        plt.ylim((0, 1))
        plt.title('')
        return ax

    def plot_scatter_n_accuracy_joint(self, data_objects, labels, label_self, markers):
        """Make plot from this and other data objects.

        Args:
            data_objects ([Data]): Other Data objects to include in plot.
            labels ([str]): Labels to use for Data_objects.
            label_self (str): Label to use for this Data object.

        Returns: Axis object.

        """
        dataframes = [self.df] + [data.df for data in data_objects]
        labels = [label_self] + labels

        acc = []
        n = []
        statistics = []
        for df, label in zip(dataframes, labels):
            acc = df.groupby('worker')['correct'].mean()
            n = df.groupby('worker')['question'].count()
            df_new = pd.concat([acc, n], axis=1)
            df_new['dataset'] = label
            statistics.append(df_new)

        df = pd.concat(statistics, axis=0)
        sns.lmplot('question', 'correct', data=df, hue='dataset',
                   markers=markers, fit_reg=False)
        plt.xlabel('Number of questions answered')
        plt.ylabel('Accuracy')
        plt.xlim((0, None))
        plt.ylim((0, 1))
        plt.title('')
        return plt.gca()

    def plot_rolling_mean_accuracy(self, window=10):
        for w, df in self.df.groupby('worker'):
            df = df.reset_index(drop=True)
            plt.plot(df.index, pd.rolling_mean(df['correct'], window=window),
                     color='blue', alpha=0.1)
        plt.ylim(0.5, 1)
        plt.xlim(0, None)
        plt.xlabel('Number of questions answered')
        plt.ylabel('Rolling accuracy ({} question window)'.format(window))
        return plt.gca()

    def plot_rolling_mean_accuracy_ts(self, window=20):
        ts = []
        for w, df in self.df.groupby('worker'):
            df = df.reset_index(drop=True)
            if 'condition' in df:
                condition = df['condition']
            else:
                condition = 'all'
            rolling_acc = pd.rolling_mean(df['correct'], window=window)
            rolling_acc.name = 'rolling_accuracy'
            df = rolling_acc.reset_index().rename(columns={'index': 't'})
            df['condition'] = condition
            df['worker'] = w
            ts.append(df)
        ax, df_stat = tsplot_robust(pd.concat(ts, axis=0), time='t',
                                    condition='condition', unit='worker',
                                    value='rolling_accuracy', ci=95)
        plt.ylim(0.5, 1)
        plt.xlim(0, None)
        plt.xlabel('Number of questions answered')
        plt.ylabel('Mean rolling accuracy ({} question window)'.format(window))


class TeachData(Data):
    """Class for retrieving and plotting Teach Data."""

    def __init__(self, df):
        """Initialize.

        Args:
            df:     Must have 'worker', 'question', 'gt', 'answer',
                    and 'correct' columns. Optionally, has 'time'.

        """
        self.df = df
        if 'time' in self.df.columns:
            self.time = True
            self.df = self.df.sort('time')
        else:
            self.time = False

    #--------- Factory methods. -------------
    @classmethod
    def from_bragg_teach(cls, data_dir=BRAGG_TEACH_DIR, min_questions=None, conditions=None):
        """Return object from teaching data."""
        df = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        df = df.rename(columns={
            'questionid': 'question',
            'workerid': 'worker',
            'observationlabel': 'answer',
            'observationvalue': 'answertype',
            'questiongold': 'gt',
            'observationtime': 'time',
            'finalobservation': 'final',
        })
        df['gt'] = df['gt'].map(json.loads)
        df['answer'] = df['answer'].map(json.loads)
        df['condition'] = df['condition'].map(json.loads)
        df['condition'] = df['condition'].map(lambda d: d['policy']['n'])
        df['correct'] = df['gt'] == df['answer']

        if min_questions is not None:
            df = pd.concat([df for worker, df in df.groupby(
                'worker') if df.question.nunique() >= min_questions])
        if conditions is not None:
            df = df[df.condition.isin(conditions)]
        return cls(df[['question', 'worker', 'answer', 'answertype', 'gt', 'time', 'correct', 'condition', 'action', 'final']])

    def n_observations(self, final):
        """Number of observations, by condition."""
        if final:
            df = self.df[self.df.final]
        else:
            df = self.df
        for worker, df_worker in df.groupby('worker'):
            exp_ack_rows = (df_worker.action == 'exp') & (
                df_worker.answertype == 'ack')
            exp_redo_rows = (df_worker.action == 'exp') & (
                df_worker.answertype == 'label')
            ask_rows = df_worker.action == 'ask'
            ask_final_rows = df_worker.final & (df_worker.action == 'ask')
            df.ix[exp_ack_rows.index, 'cum_exp_ack'] = exp_ack_rows.cumsum()
            df.ix[exp_redo_rows.index, 'cum_exp_redo'] = exp_redo_rows.cumsum()
            df.ix[ask_rows.index, 'cum_ask'] = ask_rows.cumsum()
            df.ix[ask_final_rows.index, 'question_num'] = ask_final_rows.cumsum()
        for observation in ['ask', 'exp_ack', 'exp_redo']:
            column_name = 'cum_{}'.format(observation)
            cum_observation_max = df.groupby(['worker', 'condition', 'question_num'])[
                column_name].max().reset_index()
            ax, df_stat = tsplot_robust(cum_observation_max, time='question_num',
                                        condition='condition', unit='worker',
                                        value=column_name, ci=95)
            plt.xlabel('Number of sentences seen')
            ylabel = 'Number of {} observations'.format(observation)
            if final:
                ylabel += ' (final)'
            plt.ylabel(ylabel)
            yield observation, ax

    def make_plots(self, directory):
        """Make plots."""
        ensure_dir(directory)
        for observation, ax in self.n_observations(final=False):
            plt.savefig(os.path.join(
                directory, '{}__.png'.format(observation)))
            plt.close()
        for observation, ax in self.n_observations(final=True):
            plt.savefig(os.path.join(
                directory, '{}_final__.png'.format(observation)))
            plt.close()


def add_question_num(df, worker_column='workerid', final_column='finalobservation'):
    for worker, df_worker in df.groupby(worker_column):
        ask_final_rows = df_worker[final_column] & (df_worker.action == 'ask')
        df.ix[ask_final_rows.index, 'question_num'] = ask_final_rows.cumsum()
    df['question_num'] = df.question_num.astype(int)


def get_feedback(data):
    df = pd.read_csv(data)
    df['condition'] = df['condition'].map(
        lambda d: json.loads(d)['policy']['n'])
    add_question_num(df)
    df_feedback = df[df.feedback.notnull()]
    return df_feedback[['questionid',
                        'question_num',
                        'questiontxt',
                        'questionentityper',
                        'questionentityloc',
                        'questiongold',
                        'observationlabel',
                        'feedback',
                        'workerid',
                        'condition',
                        'action'
                        ]]


def get_feedback_bragg_teach():
    """Get feedback for bragg-teach dataset and save to feedback.csv"""
    get_feedback(
        os.path.join(BRAGG_TEACH_DIR, 'data.csv')).to_csv('feedback.csv')


def make_bragg_teach_plots():
    """Make all bragg-teach plots."""
    data = Data.from_bragg_teach(min_questions=40, conditions=[3, 20])
    data.make_plots('bragg-teach')
    data.make_data('bragg-teach.csv')
    data = Data.from_bragg_teach(conditions=[3, 20])
    data.make_plots('bragg-teach_no_min')
    for relation in ['died in', 'born in', 'has nationality', 'lived in', 'traveled to']:
        data = Data.from_bragg_teach(min_questions=40, conditions=[
                                     3, 20], relations=[relation])
        data.make_plots('bragg-teach-{}'.format(relation))
    data = TeachData.from_bragg_teach(min_questions=40, conditions=[3, 20])
    data.make_plots('bragg-teach')


def make_other_plots():
    for worker_type in ['ordinary', 'normal', 'master', None]:
        name = 'rajpal'
        if worker_type is not None:
            name += '-' + worker_type
        data = Data.from_rajpal_icml15(worker_type=worker_type)
        data.make_plots(name)
        data.make_data('{}.csv'.format(name))

    data = Data.from_bragg_hcomp13(positive_only=False)
    data.make_plots('bragg')
    data.make_data('bragg.csv')

    data = Data.from_bragg_hcomp13(positive_only=True)
    data.make_plots('bragg-pos')
    data.make_data('bragg-pos.csv')

    data = Data.from_lin_aaai12(workflow='tag')
    data.make_plots('lin-tag')
    data.make_data('lin-tag.csv')

    data = Data.from_lin_aaai12(workflow='wiki')
    data.make_plots('lin-wiki')
    data.make_data('lin-wiki.csv')

def make_aamas_plot():
    """Make scatter plot with all datasets."""
    markers = ['x','o','v']
    data1 = Data.from_rajpal_icml15(worker_type=None)
    data2 = Data.from_lin_aaai12(workflow='tag')
    data3 = Data.from_lin_aaai12(workflow='wiki')
    ax = data1.plot_scatter_n_accuracy_joint(
        [data2, data3], labels=['LinTag', 'LinWiki'], label_self='Rajpal',
        markers=markers)
    plt.savefig('aamas_new_plot.png')


if __name__ == '__main__':
    make_bragg_teach_plots()
    make_other_plots()
    make_aamas_plot()
