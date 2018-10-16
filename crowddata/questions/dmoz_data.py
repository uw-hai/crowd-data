"""Datasets from DMOZ"""
import os
import util

def make_travel_dataset():
    """Dataset with travel pages from DMOZ.

    Reference:
    Kulesza, Todd, et al. 2014.
    Structured labeling for facilitating concept evolution in machine learning.
    In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems.

    """
    util.make_dataset_dmoz(
        n_items=600,
        seed=0,
        category_parent='Top/Recreation/Travel',
        rawdir=os.environ['DMOZ_TRAVEL_RAW'],
        webdir=os.environ['DMOZ_TRAVEL_WEB'],
    )
