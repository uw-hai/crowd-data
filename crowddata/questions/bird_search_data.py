import random
import os
import urllib
import csv
import util
import json


def make_house_sparrow_or_sparrow(
        seed=None,
        rawdir=None,
        webdir=os.environ['SPARROW_FLICKR_WEB']):
    """Make house sparrow or sparrow task.

    Uses Flickr API.

    Args:
        seed: Hashable seed.
        rawdir (str): Directory to store results before randomization.
        webdir (str): Directory to store randomized and normalized task.

    """
    if seed is not None:
        random.seed(seed)
    options = dict(per_page=100, content_type=1,
                   sort='relevance', query_type='text', pages=(1, 2))
    sparrows = util.get_flickr_photos(query='sparrow', **options)
    house_sparrows = util.get_flickr_photos(query='house sparrow', **options)
    for i, sparrow in enumerate(sparrows):
        sparrow['query'] = 'sparrow'
        sparrow['result_ind'] = i

        # Write intermediary results.
        if rawdir is not None:
            util.ensure_dir(os.path.join(rawdir, 'images', 'sparrow'))
            sparrow['rawpath'] = 'images/sparrow/{}.jpg'.format(sparrow['id'])
            urllib.urlretrieve(sparrow['url'], os.path.join(
                rawdir, sparrow['rawpath']))
    for i, sparrow in enumerate(house_sparrows):
        sparrow['query'] = 'house sparrow'
        sparrow['result_ind'] = i

        # Write intermediary results.
        if rawdir is not None:
            util.ensure_dir(os.path.join(rawdir, 'images', 'house_sparrow'))
            sparrow[
                'rawpath'] = 'images/house_sparrow/{}.jpg'.format(sparrow['id'])
            urllib.urlretrieve(sparrow['url'], os.path.join(
                rawdir, sparrow['rawpath']))

    # Write more intermediary results.
    if rawdir is not None:
        with open(os.path.join(rawdir, 'query_sparrow.csv'), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=sparrows[0].keys())
            writer.writerows(sparrows)
        with open(os.path.join(rawdir, 'query_house_sparrow.csv'), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=house_sparrows[0].keys())
            writer.writerows(house_sparrows)

    all_sparrows = sparrows + house_sparrows
    random.shuffle(all_sparrows)

    webdir_images = os.path.join(webdir, 'images')
    util.ensure_dir(webdir_images)

    sparrows_out = []
    for i, sparrow in enumerate(all_sparrows):
        d = dict(id=i, data=sparrow)
        d['data']['path'] = 'images/{}.jpg'.format(d['data']['id'])
        urllib.urlretrieve(d['data']['url'], os.path.join(
            webdir, d['data']['path']))
        sparrows_out.append(d)
    with open(os.path.join(webdir, 'data.json'), 'w') as f:
        json.dump(dict(data=sparrows_out), f)


def make_bird_subclusters():
    """Task with ambiguous bird clusters."""
    base_query = {'safeSearch': 'Strict'}
    queries = [{'query': 'bird', 'imageType': 'AnimatedGif'},
               {'query': 'bird', 'imageType': 'Clipart'},
               {'query': 'bird', 'imageType': 'Line'},
               {'query': 'bird', 'imageType': 'Photo'},
               {'query': 'bird mascot', 'imageType': 'Photo'},
               {'query': 'bird flock', 'imageType': 'Photo'},
               {'query': 'bird art', 'imageType': 'Photo'},
               {'query': 'extinct bird', 'imageType': 'Photo'},
               {'query': 'lego bird', 'imageType': 'Photo'},
               {'query': 'bird tattoo', 'imageType': 'Photo'},
               {'query': 'bird egg', 'imageType': 'Photo'},
               {'query': 'cooked poultry', 'imageType': 'Photo'},
               {'query': 'bird diagram', 'imageType': 'Photo'},
               {'query': 'people feeding birds', 'imageType': 'Photo'},
               ]
    for query in queries:
        query.update(base_query)

    util.make_subclusters(
        queries=queries, rawdir=os.environ['BIRD_OR_NOT_RAW'],
        webdir=os.environ['BIRD_OR_NOT_WEB'], n_images=20,
        query_indices=[0, 1, 3], seed=0)

def make_birds_flickr_dataset():
    """Datasets called birds_flickr_1 and birds_flickr_2"""
    make_house_sparrow_or_sparrow(
        seed=0, rawdir=os.environ['SPARROW_FLICKR_RAW'],
        webdir=os.environ['SPARROW_FLICKR_WEB'])

def make_bird_or_not_dataset():
    """Dataset called bird_or_not"""
    make_bird_subclusters()
