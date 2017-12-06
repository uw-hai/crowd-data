import csv
import copy
import os
import uuid
import logging
import math
import errno
import signal
import random
import urllib
import urllib2
import urlparse
import json
import collections
import subprocess
import shutil
import xml.etree.ElementTree as ET

import bs4

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


def _create_aggregate_rawdir_json(rawdir):
    """Create aggregate json for rawdir.

    Assumes rawdir is structured as:
        {rawdir}/{subdir}/images.json
        {rawdir}/{subdir}/{image_path}*

    """
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


def _create_webdir_task(rawdir, webdir, query_dirnames, seed=None, max_size=None):
    """Transform rawdir subdirectories into webdir task.

    Uses round-robin until max_size is reached.

    Assumes rawdir is structured as:
        {rawdir}/{query_dirname}/images.json
        {rawdir}/{query_dirname}/{image_path}*

    Args:
        rawdir (str): Path of input directory.
        webdir (str): Path of output directory.
        query_dirnames ([str]): List of subdirectories.
        seed (Optional[Object]): Randomization seed. Defaults to None.
            Randomization happens even when seed is None.

    """
    if max_size is None:
        max_size = float('inf')
    ensure_dir(os.path.join(webdir, 'images'))
    task_images = dict()
    for query_name in query_dirnames:
        subdir = os.path.join(rawdir, query_name)
        with open(os.path.join(subdir, 'images.json'), 'r') as fp:
            images = json.load(fp)
            task_images[query_name] = images
    if seed is not None:
        random.seed(seed)
    for query_name in task_images:
        random.shuffle(task_images[query_name])
    images_out = []
    index = 0
    while len(images_out) < max_size and any(len(x) > 0 for x in task_images.itervalues()):
        query_name = query_dirnames[index % len(query_dirnames)]
        subdir = os.path.join(rawdir, query_name)
        next_image = None
        while next_image is None and len(task_images[query_name]) > 0:
            candidate_image = task_images[query_name].pop()
            candidate_image_fullpath = os.path.join(
                subdir, candidate_image['path'],
            )
            if os.path.exists(candidate_image_fullpath):
                next_image = candidate_image
                images_out.append(next_image)
                shutil.copy(candidate_image_fullpath,
                            os.path.join(webdir, 'images'))
        index += 1
    random.shuffle(images_out)
    images_out = [dict(data=image, id=i) for i, image in enumerate(images_out)]
    with open(os.path.join(webdir, 'data.json'), 'w') as fp:
        json.dump(dict(data=images_out), fp)


def _get_web_driver(window_size=(1024, 768), timeout=None):
    """Get new PhantomJS driver instance."""
    from selenium import webdriver
    driver = webdriver.PhantomJS()
    driver.set_window_size(*window_size)
    if timeout is not None:
        driver.set_page_load_timeout(timeout)
    return driver


def _write_query_subdir(images, rawdir, query_dirname, query=None,
                        max_images=None, timeout=None, screenshot=False,
                        driver=None):
    import selenium.common.exceptions as sce
    if max_images is None:
        max_images = float('inf')
    query_subdir = os.path.join(rawdir, query_dirname)
    json_path = os.path.join(query_subdir, 'images.json')
    if not os.path.exists(json_path):
        ensure_dir(os.path.join(query_subdir, 'images'))
        images_succeeded = []
        for image in images:
            if len(images_succeeded) == max_images:
                break
            if image['encodingFormat'] == 'animatedgif':
                ext = 'gif'
            else:
                ext = image['encodingFormat']
            fname = '{}.{}'.format(image['imageId'], ext)
            outpath = os.path.join(query_subdir, 'images', fname)
            if not screenshot:
                # urllib doesn't follow redirects properly, so use wget.
                wget_extra_args = [
                    '--header', 'Accept: text/html',
                    '--user-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0',
                    '--tries', '3']
                if timeout is not None:
                    wget_extra_args += ['--timeout', str(timeout)]
                try:
                    subprocess.check_output(
                        ['wget', image['contentUrl'], '-O', outpath] + wget_extra_args)
                    images_succeeded.append(
                        dict(path='images/{}'.format(fname),
                             bingdata=image,  # Called bingdata for historical reasons.
                             query=query))
                except subprocess.CalledProcessError as e:
                    print e.output
            else:
                try:
                    if driver is None:
                        cur_driver = _get_web_driver(timeout=timeout)
                    else:
                        cur_driver = driver
                    cur_driver.get(image['contentUrl'])
                    cur_driver.save_screenshot(outpath)
                    images_succeeded.append(
                        dict(path='images/{}'.format(fname),
                             bingdata=image,  # Called bingdata for historical reasons.
                             query=query))
                except (urllib2.URLError, sce.TimeoutException) as e:
                    print e
                finally:
                    if driver is None:
                        cur_driver.quit()

        with open(json_path, 'w') as fp:
            json.dump(images_succeeded, fp)
    #if screenshot and driver is None:
    #    # cur_driver.service.process.send_signal(signal.SIGTERM)
    #    cur_driver.quit()
    #    #try:
    #    #    cur_driver.quit()
    #    #except OSError as e:
    #    #    if e.errno != errno.EBADF:
    #    #        raise


def test():
    driver = _get_web_driver()
    driver.get('http://www.bridgeholidaysfrance.com')
    driver.save_screenshot('test.png')
    driver.quit()


def make_subclusters_bing(queries, n_images, rawdir, webdir,
                          timeout=None, seed=None):
    """Make subclusters from Bing image searches.

    Args:
        queries ([dict]): List of query arguments.
        n_images ([int]): Number of results per query.
        rawdir (str): Directory to store raw partial results. Each subquery
            will create rawdir/{QUERY}/images.json and images of the form
            rawdir/{QUERY}/images/*.{gif,png,jpeg}.
        webdir (str):
        timeout (Optional[int]): Seconds until timeout.
        seed: Seed for randomization of final dataset.

    """
    n_images_list = n_images
    query_dirnames = [dict_to_str(query) for query in queries]
    # Assume wget does not timeout more than half of time.
    for query, query_name, n_images in zip(queries, query_dirnames, n_images_list):
        n_images_safe = int(round(n_images * 1.5))
        images = get_bing_images(n_images=n_images_safe, offset=0, **query)
        _write_query_subdir(
            images=images,
            rawdir=rawdir,
            query_dirname=query_name,
            query=query,
            max_images=n_images,
            timeout=timeout)

    _create_aggregate_rawdir_json(rawdir)  # Optional, for web view
    _create_webdir_task(rawdir, webdir, query_dirnames, seed)


def _find_imagenet_synsets(query, query_prefix='http://image-net.org/search?'):
    """Find all synset IDs with query term."""
    url = query_prefix + urllib.urlencode({'q': query})
    response = urllib2.urlopen(url).read().decode('utf-8')

    wnids = set()
    soup = bs4.BeautifulSoup(response)
    for node in soup.select('td > a'):
        if node.children.next().name == 'span':
            href = node['href']
            wnid = urlparse.parse_qs(urlparse.urlparse(href).query)['wnid'][0]
            wnids.add(wnid)
    return list(wnids)


def _find_imagenet_urls(wnid, query_prefix='http://www.image-net.org/api/text/imagenet.synset.geturls?'):
    """Get the URLs associated with a synset."""
    url = query_prefix + urllib.urlencode({'wnid': wnid})
    response = urllib2.urlopen(url).read().decode('utf-8')
    soup = bs4.BeautifulSoup(response)
    return soup.string.split()


def _get_imagenet_synset_words(wnid, query_prefix='http://www.image-net.org/api/text/wordnet.synset.getwords?'):
    """Get the terms associated with a synset."""
    url = query_prefix + urllib.urlencode({'wnid': wnid})
    response = urllib2.urlopen(url).read().decode('utf-8')
    soup = bs4.BeautifulSoup(response)
    return [s.strip() for s in soup.string.split('\n') if s]


def _query_from_wnid(wnid, download_words=True):
    """Get a dictionary representing a wnid."""
    query = {'wnid': wnid}
    if download_words:
        query['words'] = _get_imagenet_synset_words(wnid)
    return query


def _download_imagenet_images(wnid, rawdir, username, accesskey):
    download_url = 'http://www.image-net.org/download/synset?wnid={wnid}&username={username}&accesskey={accesskey}&release=latest&src=stanford'.format(
        wnid=wnid, username=username, accesskey=accesskey)
    query_subdir = os.path.join(rawdir, wnid)
    json_path = os.path.join(query_subdir, 'images.json')
    if not os.path.exists(json_path):
        image_subdir = os.path.join(query_subdir, 'images')
        ensure_dir(image_subdir)
        try:
            tarpath = os.path.join(image_subdir, '{0}.tar'.format(wnid))
            subprocess.check_output(
                ['wget', download_url, '-O', tarpath])
            subprocess.check_output(
                ['tar', '-xf', tarpath, '-C', image_subdir])
            subprocess.check_output(
                ['rm', tarpath])
            images = os.listdir(image_subdir)
            with open(json_path, 'w') as fp:
                json.dump(
                    [dict(path='images/{}'.format(fname),
                          bingdata=None,  # Called bingdata for historical reasons.
                          query=_query_from_wnid(wnid)) for
                     fname in images],
                    fp)
        except subprocess.CalledProcessError as e:
            print e.output


def make_dataset_imagenet(rawdir, webdir, query, max_size,
                          max_subcategory_frac=0.1, seed=None, timeout=None,
                          imagenet_username=None, imagenet_accesskey=None):
    """Make a dataset from ImageNet.

    Args:
        query_term (str):
        max_size (int):
        max_subcategory_frac (float):

    """
    if seed is not None:
        random.seed(seed)
    synsets = _find_imagenet_synsets(query)
    images = dict()
    for wnid in synsets:
        if imagenet_username is not None and imagenet_accesskey is not None:
            _download_imagenet_images(
                wnid=wnid,
                rawdir=rawdir,
                username=imagenet_username,
                accesskey=imagenet_accesskey)
        else:
            urls = _find_imagenet_urls(wnid)
            random.shuffle(urls)
            images = [
                dict(imageId=str(uuid.uuid4()),
                     encodingFormat=os.path.splitext(urlparse.urlparse(url).path)[1][1:],
                     contentUrl=url,
                     wnid=wnid) for
                i, url in enumerate(urls)]
            _write_query_subdir(
                images=images,
                rawdir=rawdir,
                query_dirname=wnid,
                query=_query_from_wnid(wnid),
                max_images=round(max_size * max_subcategory_frac),
                timeout=timeout)

    _create_aggregate_rawdir_json(rawdir)  # Optional, for web view
    _create_webdir_task(
        rawdir=rawdir,
        webdir=webdir,
        query_dirnames=synsets,
        seed=seed,
        max_size=max_size)

#--------- DMOZ-------
def _get_dmoz_urls(category_parent, dmoz_path, child_depth):
    """Return mapping from category to list of urls.

    Directions for obtaining data for dmoz_path:
    Reference:
    > Sood, Gaurav, 2016, "Parsed DMOZ data", doi:10.7910/DVN/OMV93V, Harvard Dataverse, V4, UNF:6:gLmkKDEXzZ0xV09qlqzWdQ==

    Obtain p7zip and run these commands:
    > wget https://dataverse.harvard.edu/api/access/datafile/2841558
    > 7z e 2841558

    Add a line like the following to your ~/.bashrc:
    > export DMOZ=/path/to/parsed-subdomain.csv

    """
    categories = collections.defaultdict(list)
    with open(dmoz_path) as f:
        reader = csv.reader(f)
        for line in reader:
            url = line[0]
            for category in line[1:]:
                if category.startswith(category_parent):
                    subcategories_in_parent = len(category_parent.split('/'))
                    subcategories = category.split('/')[
                        subcategories_in_parent:(subcategories_in_parent + child_depth)
                    ]
                    shortened_category = '/'.join(
                        category_parent.split('/') + subcategories
                    )
                    categories[shortened_category].append({
                        'category': category,
                        'category_short': shortened_category,
                        'url': url,
                    })
                    break
    return categories


def make_dataset_dmoz(n_items, rawdir, webdir, category_parent,
                      dmoz_path=os.environ.get('DMOZ'), child_depth=2,
                      timeout=30, seed=None, crop=True):
    """Make subclusters from DMOZ categories.

    Args:
        categories ({str: [dict]}): Webpages by category.
        n_items (int): Number of results per query.
        rawdir (str): Directory to store raw partial results. Each subquery
            will create rawdir/{CATEGORY}/images.json and images of the form
        webdir (str):
        timeout (Optional[int]): Seconds until timeout.
        seed (Optional[Object]): Seed for randomization of final dataset.

    """
    categories = _get_dmoz_urls(category_parent, dmoz_path, child_depth)
    shuffled_categories = categories.keys()
    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled_categories)
    data = collections.defaultdict(list)
    N = n_items * 1.2  # Get 20% extra for padding.
    remaining_categories = copy.deepcopy(categories)
    while sum(len(x) for x in data.itervalues()) < N:
        for category in shuffled_categories:
            urls = remaining_categories[category]
            random.shuffle(urls)
            if len(urls) >= 1:
                url = urls[0]['url']
                full_url = url
                if full_url.startswith('.'):
                    full_url = 'www' + full_url
                full_url = 'http://' + full_url
                data[category].append({
                    'imageId': str(uuid.uuid4()),
                    'encodingFormat': 'png',
                    'contentUrl': full_url,
                    'originalUrl': urls[0]['url'],
                })
                remaining_categories[category] = remaining_categories[category][1:]
            if sum(len(x) for x in data.itervalues()) == N:
                break

    #driver = _get_web_driver(timeout=timeout)
    #driver = None
    query_dirnames = []
    for category in data:
        query_dirname = category.replace('/', '_')
        _write_query_subdir(
            query={
                'category_short': category,
                'category': categories[category][0]['category'],
            },
            images=data[category],
            rawdir=rawdir,
            query_dirname=query_dirname,
            screenshot=True,
            #driver=driver,
            timeout=timeout,
        )
        query_dirnames.append(query_dirname)
    #driver.quit()

    _create_aggregate_rawdir_json(rawdir)  # Optional, for web view
    _create_webdir_task(rawdir, webdir, query_dirnames, seed, max_size=n_items)
    if crop:
        _crop(os.path.join(webdir, 'data.json'))


def _crop(filepath, outpath=None, max_height=3072, out_format='jpg'):
    """Crop dataset to a maximum height."""
    new_data = []
    with open(filepath, 'r') as f:
        data = json.load(f)['data']
        for item in data:
            new_path = '{}.{}'.format(
                os.path.splitext(item['data']['path'])[0], out_format,
            )
            try:
                subprocess.check_output([
                    'convert',
                    os.path.join(os.path.dirname(filepath), item['data']['path']),
                    '-crop',
                    'x{}+0+0'.format(max_height),
                    '+repage',
                    os.path.join(os.path.dirname(filepath), new_path),
                ])
                item['data']['path'] = new_path
                new_data.append(item)
            except subprocess.CalledProcessError as e:
                print e
    if outpath is None:
        outpath = filepath
    with open(outpath, 'w') as f:
        json.dump({'data': new_data}, f)


def partition(filepath, sizes, priorities=None, seed=None):
    """Partitions a datafile by query.

    Args:
        filepath (str): Path to datafile.
        sizes ([int]): Partition sizes. The last entry may be -1 to indicate
            that the partition should expand to include the remaining items.
        priorities (Optional[[int]]): Specify priorities of partitions.
            Lower is higher priority.
        seed (Optional[Object]): Randomization seed.

    Side effects:
        Writes files {name_{}.ext} where {} is the index of the partition.

    """
    if seed is not None:
        random.seed(seed)
    with open(filepath, 'r') as f:
        data = json.load(f)['data']
        queries = collections.defaultdict(list)
        for item in data:
            queries[json.dumps(item['data']['query'], sort_keys=True)].append(item)
        for query in queries:
            random.shuffle(queries[query])

    if sizes[-1] == -1:
        sizes[-1] = len(data) - sum(sizes[:-1])
    elif sum(sizes) < len(data):
        sizes.append(len(data) - sum(sizes))

    keys = queries.keys()
    random.shuffle(keys)
    splits = collections.defaultdict(list)
    while (
        any(len(splits[i]) < size for i, size in enumerate(sizes))
        and any(len(x) for x in queries.itervalues())
    ):
        for query in keys:
            items = queries[query]
            indices = range(len(sizes))
            random.shuffle(indices)
            if priorities is not None:
                indices = sorted(indices, key=lambda i: priorities[i])
            for i in indices:
                if len(splits[i]) < sizes[i]:
                    try:
                        splits[i].append(items.pop())
                    except IndexError:
                        pass
    for i in xrange(len(sizes)):
        path = '{}_{}{}'.format(
            os.path.splitext(filepath)[0],
            i,
            os.path.splitext(filepath)[1],
        )
        with open(path, 'w') as f:
            json.dump({'data': splits[i]}, f)
