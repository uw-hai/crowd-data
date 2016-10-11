"""relation_data.py"""
# TODO: Randomize gold relations and structuredlabeling? (with seed)
import os
import re
import csv
import json
import collections
import random
from . import relation_data_helper as rdh
from . import util


webdir = os.path.join(os.environ['RELATION_WEB'])


def make_gold_dataset():
    data = []
    gold1 = rdh.get_relation_data('gold1_soderland')
    for relation_context in gold1.data:
        excerpt = relation_context.excerpt.__dict__
        entities = [entity.__dict__ for entity in relation_context.entities]
        cls = None
        for annotation in relation_context.annotations:
            if annotation.relation == 'lived in':
                cls = annotation.value
        data.append(dict(cls=cls, data=dict(
            excerpt=excerpt, entities=entities)))

    for ind, it in enumerate(data):
        it['id'] = ind

    websubdir = os.path.join(webdir, 'gold1')
    util.ensure_dir(websubdir)
    with open(os.path.join(websubdir, 'data.json'), 'w') as f:
        json.dump(dict(data=data), f)


def make_train_dataset(start_ind_pos, end_ind_pos, start_ind_neg, end_ind_neg,
                       relation, seed=None):
    """Make training data.

        Args:
            start_ind_pos (int): Starting index into positive data.
            end_ind_pos (int): Ending index into positive data.
            start_ind_neg (int): Starting index into negative data.
            end_ind_neg (int): Ending index into negative data.
            relation (str): Relation name. Can be one of
                ['travel', 'lived', 'born', 'died', 'nationality'].

    """
    if seed is not None:
        random.seed(seed)
    src_dir = os.environ['RELATION_TRAINING_DIR']
    files = dict()
    files['pos'] = os.path.join(
        src_dir, 'train_DS_pos_new_feature_{}'.format(relation))
    files['neg'] = neg_file = os.path.join(
        src_dir, 'train_DS_neg_new_feature_{}'.format(relation))
    data = collections.defaultdict(list)
    indices = dict(pos=(start_ind_pos, end_ind_pos),
                   neg=(start_ind_neg, end_ind_neg))
    for cls in files:
        with open(files[cls], 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i >= indices[cls][0] and i < indices[cls][1]:
                    entities = [
                        {'txt': row[0],
                         'i0': int(row[1]),
                         'i1': int(row[2]),
                         'type_': 'per'},
                        {'txt': row[3],
                         'i0': int(row[4]),
                         'i1': int(row[5]),
                         'type_': 'loc'}]
                    excerpt = {'docid': row[6],
                               'txt': row[11]}
                    # Convert word indices to character.
                    matches = list(re.finditer(r'[^ ]+', excerpt['txt']))
                    for entity in entities:
                        entity['i0'] = matches[entity['i0']].start()
                        entity['i1'] = matches[entity['i1']].end()
                    data[cls].append(dict(data=dict(entities=entities,
                                                    excerpt=excerpt,
                                                    srcid=i),
                                          cls=-1 if cls=='neg' else 1))
    data_out = data['pos'] + data['neg']
    random.shuffle(data_out)
    for i, d in enumerate(data_out):
        d['id'] = i
    websubdir = os.path.join(webdir, 'train_{}_pos_{}_{}_neg_{}_{}'.format(
        relation, start_ind_pos, end_ind_pos, start_ind_neg, end_ind_neg))
    util.ensure_dir(websubdir)
    with open(os.path.join(websubdir, 'data.json'), 'w') as f:
        json.dump(dict(data=data_out), f)


if __name__ == '__main__':
    make_gold_dataset()
    make_train_dataset(0, 250, 0, 250, 'lived', seed=0)
