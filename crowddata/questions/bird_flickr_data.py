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


def get_flickr_photos(query, pages, api_key=os.environ['FLICKR_API_KEY'], per_page=100):
    """Get photo details from Flickr for the given query.

    Args:
        query (str): Search query.
        pages ((int, int)): Start and end page.
        api_key (str): Flickr API key.
        per_page (int): Number of results to return per page. Max is 500.

    """
    n_pages = None
    pages = (pages[0], pages[1] + 1)
    data_out = []
    for page in xrange(*pages):
        logging.info('Querying page {} of {}'.format(page, pages[1]))
        if n_pages is not None and page > n_pages:
            break
        data = dict(api_key=api_key, query=urllib2.quote(query), per_page=per_page, page=page)
        res = urllib2.urlopen(
            'https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key={api_key}&text={query}&page={page}'.format(**data))
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

def make_house_sparrow_or_sparrow(seed=None):
    if seed is not None:
        random.seed(seed)
    sparrows = get_flickr_photos(query='sparrow', pages=(1, 2))
    house_sparrows = get_flickr_photos(query='house sparrow', pages=(1, 2))
    for i, sparrow in enumerate(sparrows):
        sparrow['query'] = 'sparrow'
        sparrow['result_ind'] = i
    for i, sparrow in enumerate(house_sparrows):
        sparrow['query'] = 'house sparrow'
        sparrow['result_ind'] = i
    all_sparrows = sparrows + house_sparrows
    random.shuffle(all_sparrows)

    webdir = os.environ['SPARROW_FLICKR_WEB']
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
    make_house_sparrow_or_sparrow(seed=0)
