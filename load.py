import os
import csv
import pandas as pd

RAJPAL_ICML15_DIR = '/homes/gws/jbragg/data/rajpal-icml15/'
BRAGG_HCOMP13_DIR = '/homes/gws/jbragg/data/bragg-hcomp13/hcomp13-multilabel-data/'
LIN_AAAI12_DIR    = '/homes/gws/jbragg/data/lin-aaai12/'


# TODO: Enable retrieval using workflow=None, as with Rajpal data.
def load_lin_aaai12(data_dir=LIN_AAAI12_DIR, workflow='tag'):
    """Return dataframe with joined data for lin-aaai12.

    Args:
        workflow:   Either 'tag' or 'wiki' for different dataset.

    """
    # Load answers.
    basedir = os.path.join(data_dir, 'testingData')
    files = os.listdir(basedir)
    files = [os.path.join(basedir, f) for f in files if f.startswith(workflow)]
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
    df['correct'] = df['answer'] == df['answer_gt']
    return df

def load_bragg_hcomp13(data_dir=BRAGG_HCOMP13_DIR, positive_only=False):
    df = pd.read_csv(os.path.join(data_dir, 'data.csv'))
    df_gold = pd.read_csv(os.path.join(data_dir, 'gold.csv'))
    df_gold = df_gold.rename(columns={'Unnamed: 0': 'item'}).fillna(0)
    df_gold = df_gold.set_index('item')
    df_gold = df_gold.stack()
    df_gold.index = df_gold.index.set_names(['item', 'label'])
    df_gold.name = 'gt'
    df_gold = df_gold.reset_index()
    df = df.merge(df_gold, how='left', on=['item', 'label'])
    df['correct'] = df['selected'].astype(int) == df['gt'].astype(int)

    if positive_only:
        df = df[df['gt'].astype(int) == 1]

    # Construct questions out of item-label combinations.
    df['question'] = df.item + '-' + df.label
    return df

def load_rajpal_icml15(data_dir=RAJPAL_ICML15_DIR, worker_type=None):
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
        df_gold = pd.DataFrame([{'difficulty': d[0],
                                 'gt': v[0]} for d, v in zip(difficulty, gt)])
        df_gold.index.names = ['question']
        df_gold = df_gold.reset_index()

    # Load answers without time stamps.
    df_acc = pd.DataFrame()
    for t in ('ordinary', 'master', 'normal'):
        df = pd.read_csv(os.path.join(
            data_dir, '{}PoolResultsCSV'.format(t.capitalize())), header=None)
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
    df['correct'] = df['gt'].astype(int) == df['answer'].astype(int)

    if worker_type is not None:
        return df[df.worker_type == worker_type]
    else:
        return df
