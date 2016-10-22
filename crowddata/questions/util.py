import os
import logging
import random
import urllib2
import json
import subprocess
import urllib
import shutil
import xml.etree.ElementTree as ET


BING_MAX_COUNT = 150

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def dict_to_str(json_dict):
    """Normalize single-level dictionary into string."""
    return '-'.join('_'.join([key, json_dict[key]]) for
                    key in sorted(json_dict))


def get_bing_images(query, n_images, offset=0, api_key=os.environ['BING_API_KEY'], **queryargs):
    """Get images from Bing.

    Args:
        query (str): Search query.
        n_images (int): Number of results to return.
        offset (int): Starting offset.
        api_key (str): Microsoft API key.
        queryargs (dict): Additional arguments. For more details, see
            https://msdn.microsoft.com/en-us/library/dn760791.aspx.

    Returns:
        [dict]: List of Bing image objects.

    """
    endpoint = 'https://api.cognitive.microsoft.com/bing/v5.0/images/search?{}'

    args = dict(q=query)
    args.update(queryargs)

    n = 0
    next_offset = offset
    total_estimated_matches = float('inf')
    target_offset = offset + n_images
    images = []
    while next_offset < target_offset:
        n_images_left = n_images - len(images)
        target_offset = min(next_offset + n_images_left,
                            total_estimated_matches)
        count = min(target_offset - next_offset, BING_MAX_COUNT)
        request = urllib2.Request(
            endpoint.format(urllib.urlencode(args)),
            headers={'Ocp-Apim-Subscription-Key': api_key})
        response = json.load(urllib2.urlopen(request))
        images += response['value']

        next_offset += count + response['nextOffsetAddCount']
        total_estimated_matches = response['totalEstimatedMatches']
    return images
    tmpfile = 'tmp2.json'
    with open(tmpfile, 'w') as fp:
        json.dump(images, fp)

def get_flickr_photos(query, pages, query_type='text', api_key=os.environ['FLICKR_API_KEY'], **queryargs):
    """Get photo details from Flickr for the given query.

    Args:
        query (str): Search query.
        pages ((int, int)): Start and end page.
        query_type (str): Either 'text' or 'tags'.
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
        data = dict(api_key=api_key, page=page, query_type=query)
        data.update(queryargs)
        res = urllib2.urlopen(
            'https://api.flickr.com/services/rest/?method=flickr.photos.search&{}'.format(urllib.urlencode(data)))
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

def make_subclusters(queries, rawdir, webdir, n_images, query_indices, seed=None):
    """Make subclusters from Bing image searches.

    Args:
        queries ([dict]): List of query arguments.
        rawdir (str): Directory to store raw partial results. Each subquery
            will create rawdir/{QUERY}/images.json and images of the form
            rawdir/{QUERY}/images/*.{gif,png,jpeg}.
        webdir (str):
        n_images (int): Number of images to fetch per query.
        query_indices ([int]): List of indices into queries indicating which
            subclusters to use for randomized dataset.

    """
    for query in queries:
        query_name = dict_to_str(query)
        images = get_bing_images(n_images=n_images, offset=0, **query)
        query_subdir = os.path.join(rawdir, query_name)
        json_path = os.path.join(query_subdir, 'images.json')
        if not os.path.exists(json_path):
            ensure_dir(os.path.join(query_subdir, 'images'))
            images_succeeded = []
            for image in images:
                if image['encodingFormat'] == 'animatedgif':
                    ext = 'gif'
                else:
                    ext = image['encodingFormat']
                fname = '{}.{}'.format(image['imageId'], ext)
                outpath = os.path.join(query_subdir, 'images', fname)
                # urllib doesn't follow redirects properly, so use wget.
                wget_extra_args = [
                    '--header', 'Accept: text/html', '--user-agent',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0']
                try:
                    subprocess.check_output(
                        ['wget', image['contentUrl'], '-O', outpath] + wget_extra_args)
                    images_succeeded.append(
                        dict(path='images/{}'.format(fname),
                             bingdata=image,
                             query=query))
                except subprocess.CalledProcessError as e:
                    print e.output
            with open(json_path, 'w') as fp:
                json.dump(images_succeeded, fp)

    # Create aggregate json for web view.
    query_results = []
    for subdir in os.listdir(rawdir):
        if os.path.isdir(os.path.join(rawdir, subdir)):
            with open(os.path.join(rawdir, subdir, 'images.json'), 'r') as fp:
                images = json.load(fp)
            for image in images:
                image['path'] = '{}/{}'.format(subdir, image['path'])
            query_results.append(images)
    with open(os.path.join(rawdir, 'images.json'), 'w') as fp:
        json.dump(query_results, fp)

    # Create webdir task.
    ensure_dir(os.path.join(webdir, 'images'))
    task_images = []
    for query_index in query_indices:
        subdir = os.path.join(rawdir, dict_to_str(queries[query_index]))
        with open(os.path.join(subdir, 'images.json'), 'r') as fp:
            images = json.load(fp)
            for image in images:
                shutil.copy(os.path.join(subdir, image['path']), os.path.join(webdir, 'images'))
            task_images += images
    if seed is not None:
        random.seed(seed)
    random.shuffle(task_images)
    images_out = []
    for i, image in enumerate(task_images):
        images_out.append(dict(data=image, id=i))
    with open(os.path.join(webdir, 'data.json'), 'w') as fp:
        json.dump(dict(data=images_out), fp)
