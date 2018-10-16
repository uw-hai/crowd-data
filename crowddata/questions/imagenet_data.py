"""Datasets from ImageNet"""
import os
import util

def make_car_or_not_dataset():
    """Dataset called cat_or_not.

    Reference:
    Joseph Chee Chang, Saleema Amershi, and Ece Kamar. 2017.
    Revolt: Collaborative Crowdsourcing for Labeling Machine Learning Datasets.
    In Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems (CHI '17).
    ACM, New York, NY, USA, 3180-3191. DOI: http://dx.doi.org/10.1145/3025453.3026044

    """
    util.make_dataset_imagenet(
        query='car',
        max_size=600,
        max_subcategory_frac=0.1,
        seed=0,
        rawdir=os.environ['CAR_OR_NOT_RAW'],
        webdir=os.environ['CAR_OR_NOT_WEB'],
        timeout=20,
        imagenet_username=os.environ.get('IMAGENET_USERNAME', None),
        imagenet_accesskey=os.environ.get('IMAGENET_ACCESSKEY', None))


def make_cat_or_not_dataset():
    """Dataset called cat_or_not.

    Reference:
    Joseph Chee Chang, Saleema Amershi, and Ece Kamar. 2017.
    Revolt: Collaborative Crowdsourcing for Labeling Machine Learning Datasets.
    In Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems (CHI '17).
    ACM, New York, NY, USA, 3180-3191. DOI: http://dx.doi.org/10.1145/3025453.3026044

    """
    util.make_dataset_imagenet(
        query='cat',
        max_size=600,
        max_subcategory_frac=0.1,
        seed=0,
        rawdir=os.environ['CAT_OR_NOT_RAW'],
        webdir=os.environ['CAT_OR_NOT_WEB'],
        timeout=20,
        imagenet_username=os.environ.get('IMAGENET_USERNAME', None),
        imagenet_accesskey=os.environ.get('IMAGENET_ACCESSKEY', None))
