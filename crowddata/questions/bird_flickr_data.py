import logging
import random
import os
import urllib2
import urllib
import csv
import time
import xml.etree.ElementTree as ET
import util
import json


def get_flickr_photos(pages, api_key=os.environ['FLICKR_API_KEY'], **queryargs):
    """Get photo details from Flickr for the given query.

    Args:
        pages ((int, int)): Start and end page.
        api_key (str): Flickr API key.
        queryargs (dict): Additional arguments. For more details, see
            https://www.flickr.com/services/api/flickr.photos.search.html.
            Important to use "sort=relevance", which is not default, to match
            web interface results. For example usage, see
            make_house_sparrow_or_sparrow().

    """
    n_pages = None
    pages = (pages[0], pages[1] + 1)
    data_out = []
    for page in xrange(*pages):
        logging.info('Querying page {} of {}'.format(page, pages[1]))
        if n_pages is not None and page > n_pages:
            break
        data = dict(api_key=api_key, page=page)
        data.update(queryargs)
        res = urllib2.urlopen('https://api.flickr.com/services/rest/?method=flickr.photos.search&{}'.format(urllib.urlencode(data)))
        res_string = res.read()
        root = ET.fromstring(res_string)
        photos = root.find('photos')
        n_pages = photos.attrib['pages']
        for photo in photos:
            data = dict(farm=photo.attrib['farm'],
                        server=photo.attrib['server'],
                        id=photo.attrib['id'],
                        secret=photo.attrib['secret'])
            url = 'https://farm{farm}.staticflickr.com/{server}/{id}_{secret}.jpg'.format(
                **data)
            data['url'] = url
            data['page'] = page
            data['owner'] = photo.attrib['owner']
            data['title'] = photo.attrib['title'].encode('utf8')
            data_out.append(data)
    return data_out

def make_house_sparrow_or_sparrow(
        seed=None,
        rawdir=None,
        webdir=os.environ['SPARROW_FLICKR_WEB']):
    """Make house sparrow or sparrow task.

    Args:
        seed: Hashable seed.
        rawdir (str): Directory to store results before randomization.
        webdir (str): Directory to store randomized and normalized task.

    """
    if seed is not None:
        random.seed(seed)
    options = dict(per_page=100, content_type=1, sort='relevance')
    sparrows = get_flickr_photos(pages=(1, 2), text='sparrow', **options)
    house_sparrows = get_flickr_photos(pages=(1, 2), text='house sparrow', **options)
    for i, sparrow in enumerate(sparrows):
        sparrow['query'] = 'sparrow'
        sparrow['result_ind'] = i

        # Write intermediary results.
        if rawdir is not None:
            util.ensure_dir(os.path.join(rawdir, 'images', 'sparrow'))
            sparrow['rawpath'] = 'images/sparrow/{}.jpg'.format(sparrow['id'])
            urllib.urlretrieve(sparrow['url'], os.path.join(rawdir, sparrow['rawpath']))
    for i, sparrow in enumerate(house_sparrows):
        sparrow['query'] = 'house sparrow'
        sparrow['result_ind'] = i

        # Write intermediary results.
        if rawdir is not None:
            util.ensure_dir(os.path.join(rawdir, 'images', 'house_sparrow'))
            sparrow['rawpath'] = 'images/house_sparrow/{}.jpg'.format(sparrow['id'])
            urllib.urlretrieve(sparrow['url'], os.path.join(rawdir, sparrow['rawpath']))

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
        urllib.urlretrieve(d['data']['url'], os.path.join(webdir, d['data']['path']))
        sparrows_out.append(d)
    with open(os.path.join(webdir, 'data.json'), 'w') as f:
        json.dump(dict(data=sparrows_out), f)

if __name__ == '__main__':
    make_house_sparrow_or_sparrow(seed=0, rawdir=os.environ['SPARROW_FLICKR_RAW'], webdir=os.environ['SPARROW_FLICKR_WEB'])
