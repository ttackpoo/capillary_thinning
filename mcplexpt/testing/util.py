"""
Utilities to help write the tests.
"""

import os


def get_samples_path(*paths):
    r"""
    Get the absolute path to the directory where the sample data are
    stored.

    Parameters
    ==========

    paths : iterable of str
        Subpaths under ``mcplexpt/samples/`` directory

    Returns
    =======

    path : str
        Absolute path to the sample depending on the user's system

    Examples
    ========

    >>> from mcplexpt.testing import get_samples_path
    >>> get_samples_path() # doctest: +SKIP
    'user\\mcplexpt\\samples'
    >>> get_samples_path("core", "A.txt") # doctest: +SKIP
    'user\\mcplexpt\\samples\\core\\A.txt'

    """
    this_file = os.path.abspath(__file__)
    mcplexpt_dir = os.path.join(this_file, "..", "..", "..")
    sample_dir = os.path.join(mcplexpt_dir, "samples")
    sample_dir = os.path.normpath(sample_dir)
    sample_dir = os.path.normcase(sample_dir)

    path = os.path.join(sample_dir, *paths)

    return path
