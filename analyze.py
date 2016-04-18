"""Classes to import crowdsourcing data and analyze it."""
import os
import csv
import json
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from .research_utils.util import tsplot_robust, ensure_dir


class Data(object):
    """Class for retrieving and plotting HCOMP data."""
    # TODO: Fix plotting code for data that includes more optional columns
    # than just 'time'.

    def __init__(self, df):
        """Initialize.

        - If no 'action' provided, assume 'ask' action.
        - If no 'answertype' provided, assume 'label' action.

        Args:
            df (pandas.DataFrame): Data.
                Required columns:
                    'worker', 'question', 'gt', 'answer', 'correct'
                Optional columns:
                    'condition', 'time', 'action', 'actiontype', 'answertype'

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
    def from_lin_aaai12(cls, data_dir=None, workflow='tag'):
        """Load from joined data for lin-aaai12.

        Args:
            workflow (str): Either 'tag' or 'wiki' for different dataset.

        """
        if data_dir is None:
            data_dir = os.environ['LIN_AAAI12_DIR']
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
    def from_bragg_hcomp13(cls, data_dir=None, positive_only=False):
        if data_dir is None:
            data_dir = os.environ['BRAGG_HCOMP13_DIR']
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
    def from_bragg_teach(cls, data_dir=None,
                         min_questions=None, relations=None, conditions=None):
        """Load from teaching data.

        Note:
            - Assumes 'data.csv' exists in data_dir.
            - Ignores non-final entries.
            - Includes timestamps.

        Args:
            data_dir (Optional[str]): Data directory (defaults to value set
                in BRAGG_TEACH_DIR environment variable).
            min_questions (Optional[int]): Minimum number of questions asked
                worker (defaults to no minimum).
            relations (Optional[[str]]): Relations to include (defaults to
                all relations).
            conditions (Optional[[str]]): Names of conditions. Can be
                include following conditions (None defaults to all):
                - 'pilot_10': Include teach 10 times (first study).
                - 'pilot_20': Include teach 20 times (first study).
                - 'rl_v1': Include non-parallel exp (second study).
                - 'rl_v2': Include parallel exp (second study).

        Returns:
            analyze.Data: Data object.

        """
        if data_dir is None:
            data_dir = os.environ['BRAGG_TEACH_DIR']

        df = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        df = df[df.finalobservation]  # Ignore non-final.
        df = df.rename(columns={
            'questionid': 'question',
            'workerid': 'worker',
            'observationlabel': 'answer',
            'observationvalue': 'answertype',
            'questiongold': 'gt',
            'observationtime': 'time',
            'finalobservation': 'final',
            'actionrule': 'actiontype',
        })
        df['condition'] = df['condition'].map(json.loads)
        if min_questions is not None:
            df = pd.concat([df for worker, df in df.groupby(
                'worker') if df.question.nunique() >= min_questions])

        def condition_name(condition):
            if 'n' in condition['policy'] and condition['policy']['n'] == 3:
                return 'pilot_3'
            elif 'n' in condition['policy'] and condition['policy']['n'] == 10:
                return 'pilot_10'
            elif 'n' in condition['policy'] and condition['policy']['n'] == 20:
                return 'pilot_20'
            elif ('explore_policy' in condition['policy'] and
                  condition['ask_bonus'] == 0.04):
                return 'rl_v1'
            elif ('explore_policy' in condition['policy'] and
                  condition['ask_bonus'] == 0.08):
                return 'rl_v2'
        df['condition'] = df['condition'].map(condition_name)
        if conditions is not None:
            df = df[df['condition'].isin(conditions)]

        df['gt'] = df['gt'].map(json.loads)
        df['answer'] = df['answer'].map(json.loads)
        if relations is not None:
            for column in ['gt', 'answer']:
                df[column] = df[column].map(lambda d: dict(
                    (relation, d[relation]) for relation in relations) if
                    d is not None else None)
        df['correct'] = df['gt'] == df['answer']

        df['condition'] = df['condition'].map(
            lambda x: json.dumps(x, sort_keys=True))
        return cls(df[['question', 'worker', 'answer', 'answertype', 'gt',
                       'time', 'correct', 'condition',
                       'action', 'actiontype']])

    @classmethod
    def from_rajpal_icml15(cls, data_dir=None, worker_type=None):
        """Return dataframe with joined data for rajpal-icml15.

        Args:
            worker_type:    Either 'ordinary', 'normal', 'master', or
                            None (for all worker classes).

        """
        if data_dir is None:
            data_dir = os.environ['RAJPAL_ICML15_DIR']
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


def make_bragg_teach_plots(dirname='plots'):
    """Make all bragg-teach plots."""
    options1 = {'conditions': ['pilot_3', 'pilot_20']}
    options2 = {'conditions': ['pilot_3', 'pilot_20'], 'min_questions': 40}
    options3 = {'conditions': ['rl_v1']}
    options4 = {'conditions': ['rl_v2']}
    all_options = [options1, options2, options3, options4]
    for options in all_options:
        name = '_'.join(options['conditions'])
        if 'min_questions' in options:
            name += '_min_{}_questions'.format(options['min_questions'])
        data = Data.from_bragg_teach(**options)
        data.make_plots(os.path.join(dirname, 'bragg-teach-{}'.format(name)))
        data.make_data(os.path.join(
            dirname, 'bragg-teach-{}.csv'.format(name)))
        for relation in ['died in', 'born in', 'has nationality',
                         'lived in', 'traveled to']:
            data = Data.from_bragg_teach(relations=[relation], **options)
            data.make_plots(os.path.join(
                dirname, 'bragg-teach-{}-{}'.format(name, relation)))


def make_all_plots(dirname='plots'):
    """Make plots for all datasets.

    Args:
        dirname (str): Path to output directory.

    """
    for worker_type in ['ordinary', 'normal', 'master', None]:
        name = 'rajpal'
        if worker_type is not None:
            name += '-' + worker_type
        data = Data.from_rajpal_icml15(worker_type=worker_type)
        data.make_plots(name)
        data.make_data('{}.csv'.format(name))

    data = Data.from_bragg_hcomp13(positive_only=False)
    data.make_plots(os.path.join(dirname, 'bragg'))
    data.make_data(os.path.join(dirname, 'bragg.csv'))

    data = Data.from_bragg_hcomp13(positive_only=True)
    data.make_plots(os.path.join(dirname, 'bragg-pos'))
    data.make_data(os.path.join(dirname, 'bragg-pos.csv'))

    data = Data.from_lin_aaai12(workflow='tag')
    data.make_plots(os.path.join(dirname, 'lin-tag'))
    data.make_data(os.path.join(dirname, 'lin-tag.csv'))

    data = Data.from_lin_aaai12(workflow='wiki')
    data.make_plots(os.path.join('lin-wiki'))
    data.make_data(os.path.join('lin-wiki.csv'))

    make_bragg_teach_plots(dirname=dirname)


def main():
    """Main."""
    make_all_plots()

if __name__ == '__main__':
    main()
