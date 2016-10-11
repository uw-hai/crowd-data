"""bird_data.py"""
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
    return warblers, goldfinches


def save_images_warbler_or_goldfinch(outputdir, seed=None):
    if seed is not None:
        random.seed(seed)
    warblers, goldfinches = warbler_or_goldfinch()

    warblers = [dict(cls='Warbler', data={'imageid': i}) for i in warblers]
    goldfinches = [dict(cls='Goldfinch', data={'imageid': i}) for
                   i in goldfinches]
    items = warblers + goldfinches
    random.shuffle(items)

    paths = get_imagepaths()
    for i, it in enumerate(items):
        it['id'] = i
        it['data']['path_src'] = paths[it['data']['imageid']]
        it['data']['path'] = 'images/{}.jpg'.format(i)

    images_dir = os.path.join(outputdir, 'images')
    util.ensure_dir(images_dir)

    with open(os.path.join(outputdir, 'data.json'), 'w') as f:
        json.dump(dict(data=items), f)

    with open(os.path.join(outputdir, 'stats.txt'), 'w') as f:
        f.write('Warblers: {}\n'.format(len(warblers)))
        f.write('Goldfinches: {}\n'.format(len(goldfinches)))

    for it in items:
        shutil.copy(os.path.join(rootdir, 'images', it['data']['path_src']),
                    os.path.join(outputdir, it['data']['path']))


if __name__ == '__main__':
    # print select_birds(1, 60)
    print get_superclasses()
    print dict((k, len(v)) for k, v in get_imageids().iteritems())
    save_images_warbler_or_goldfinch(os.path.expanduser(os.environ['CUB_200_2011_WEB']),
                                     seed=0)
