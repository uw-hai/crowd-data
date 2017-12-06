"""Datasets from CUB_200_2011"""
# TODO: Rename this file.
import os
import shutil
import csv
import json
import collections
import random
import numpy as np
from . import util

rootdir = os.environ['CUB_200_2011_DIR']
rootdir = os.path.join(rootdir, 'CUB_200_2011')


def get_imageids():
    labels = collections.defaultdict(list)
    with open(os.path.join(rootdir, 'image_class_labels.txt'), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            imageid = int(row[0])
            label = int(row[1])
            labels[label].append(imageid)
    return labels


def get_imagepaths():
    paths = dict()
    with open(os.path.join(rootdir, 'images.txt'), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            imageid = int(row[0])
            path = row[1]
            paths[imageid] = path
    return paths

def get_features():
    """Get features.

    certaintyids:
        - 1: 'not visible'
        - 2: 'guessing'
        - 3: 'probably'
        - 4: 'definitely'

    Returns:
        {imageid: {featureid: certaintyid}}

    """
    features = dict()
    with open(os.path.join(rootdir, 'attributes', 'image_attribute_labels.txt'), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            imageid = int(row[0])
            attributeid = int(row[1])
            certaintyid = int(row[2])
            if imageid not in features:
                features[imageid] = dict()
            features[imageid][attributeid] = certaintyid
    return features

def get_superclasses():
    """Group classes into superclasses like Finches."""
    with open(os.path.join(rootdir, 'classes.txt'), 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        groups = collections.defaultdict(list)
        for row in reader:
            label = int(row[0])
            classid = row[1]
            superclass = classid.split('.')[1].split('_')[-1]
            groups[superclass].append(label)
    return groups


def endangered_woodpecker():
    """Return endangered woodpecker task.

    Reference:
    Singla et al. Near-Optimally Teaching the Crowd to Classify. In ICML 2014.

    """
    imageids = get_imageids()
    return [{'sublabel': 'Red_cockaded_Woodpecker',
             'label': 'endangered',
             'imageids': imageids[190]},
            {'sublabel': 'Red_bellied_Woodpecker',
             'label': 'not_endangered',
             'imageids': imageids[189]},
            {'sublabel': 'Downy_Woodpecker',
             'label': 'not_endangered',
             'imageids': imageids[192]}]


def warbler_or_goldfinch():
    """Return warbler and goldfinch imageids"""
    superclasses = get_superclasses()
    imageids = get_imageids()

    warbler_classids = superclasses['Warbler']
    goldfinch_classids = superclasses['Goldfinch']

    warblers = []
    goldfinches = []
    for classid in warbler_classids:
        warblers += imageids[classid]
    for classid in goldfinch_classids:
        goldfinches += imageids[classid]
    return [{'label': 'Warbler',
             'imageids': warblers},
            {'label': 'Goldfinch',
             'imageids': goldfinches}]


def save_images(dataset, outputdir, seed=None):
    if seed is not None:
        random.seed(seed)

    if dataset == 'warbler_or_goldfinch':
        birds = warbler_or_goldfinch()
    elif dataset == 'endangered_woodpecker':
        birds = endangered_woodpecker()

    items = []
    for category in birds:
        label = category['label']
        sublabel = None if 'sublabel' not in category else category['sublabel']
        imageids = category['imageids']
        items += [dict(cls=label,
                       sublabel=sublabel,
                       data={'imageid': i}) for i in imageids]
    random.shuffle(items)

    paths = get_imagepaths()
    features = get_features()
    for i, it in enumerate(items):
        it['id'] = i
        it['data']['path_src'] = paths[it['data']['imageid']]
        it['data']['path'] = 'images/{}.jpg'.format(i)
        it['data']['features'] = features[it['data']['imageid']]

    images_dir = os.path.join(outputdir, 'images')
    util.ensure_dir(images_dir)

    with open(os.path.join(outputdir, 'data.json'), 'w') as f:
        json.dump(dict(data=items), f)

    with open(os.path.join(outputdir, 'stats.txt'), 'w') as f:
        for category in birds:
            f.write('{}: {}\n'.format(
                category['label'], len(category['imageids'])))

    for it in items:
        shutil.copy(os.path.join(rootdir, 'images', it['data']['path_src']),
                    os.path.join(outputdir, it['data']['path']))


if __name__ == '__main__':
    # print select_birds(1, 60)
    save_images('endangered_woodpecker',
                os.path.expanduser(os.environ['WOODPECKER_WEB']),
                seed=0)
    import sys
    sys.exit()
    print dict((k, len(v)) for k, v in get_imageids().iteritems())
    save_images('warbler_or_goldfinch',
                os.path.expanduser(os.environ['CUB_200_2011_WEB']),
                seed=0)
