
import zipfile

import numpy as np

import scipy.sparse as sp

from lightfm.datasets import _common


def _read_raw_data(path):
    """
    Return the raw lines of the train and test files.
    """

    with zipfile.ZipFile(path) as datafile:
        return datafile.read('ml-100k/u.data').decode().split('\n')


def _parse(data):

    for line in data:

        if not line:
            continue

        uid, iid, rating, timestamp = [int(x) for x in line.split('\t')]

        yield uid, iid, rating, timestamp


def _get_dimensions(data):

    uids = set()
    iids = set()
    timestamps = set()

    for uid, iid, _, timestamp in data:
        uids.add(uid)
        iids.add(iid)
        timestamps.add(timestamp)

    rows = max(uids) + 1
    cols = max(iids) + 1
    timestamps = max(timestamps) + 1

    return rows, cols, timestamps


def _build_interaction_matrices(rows, cols, data, min_rating):

    train_mat = sp.lil_matrix((rows, cols), dtype=np.int32)
    test_mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    uids_in_test = set()
    uids_in_train = set()

    for uid, iid, rating, timestamp in data:

        if uid not in uids_in_test and uid not in uids_in_train:
            if np.random.random() < 0.2:
                uids_in_test.add(uid)
            else:
                uids_in_train.add(uid)

            if uid in uids_in_test:
                mat = test_mat
            else:
                mat = train_mat

        if rating >= min_rating:
            mat[uid, timestamp] = iid

    return train_mat, test_mat


def fetch_movielens(data_home=None, indicator_features=True, genre_features=False,
                    min_rating=0.0, download_if_missing=True):
    """
    Fetch the `Movielens 100k dataset <http://grouplens.org/datasets/movielens/100k/>`_.

    The dataset contains 100,000 interactions from 1000 users on 1700 movies,
    and is exhaustively described in its
    `README <http://files.grouplens.org/datasets/movielens/ml-100k-README.txt>`_.

    Parameters
    ----------

    data_home: path, optional
        Path to the directory in which the downloaded data should be placed.
        Defaults to ``~/lightfm_data/``.
    indicator_features: bool, optional
        Use an [n_users, n_users] identity matrix for item features. When True with genre_features,
        indicator and genre features are concatenated into a single feature matrix of shape
        [n_users, n_users + n_genres].
    genre_features: bool, optional
        Use a [n_users, n_genres] matrix for item features. When True with item_indicator_features,
        indicator and genre features are concatenated into a single feature matrix of shape
        [n_users, n_users + n_genres].
    min_rating: float, optional
        Minimum rating to include in the interaction matrix.
    download_if_missing: bool, optional
        Download the data if not present. Raises an IOError if False and data is missing.

    Notes
    -----

    The return value is a dictionary containing the following keys:

    Returns
    -------

    train: sp.coo_matrix of shape [n_users, n_items]
         Contains training set interactions.
    test: sp.coo_matrix of shape [n_users, n_items]
         Contains testing set interactions.
    item_features: sp.csr_matrix of shape [n_items, n_item_features]
         Contains item features.
    item_feature_labels: np.array of strings of shape [n_item_features,]
         Labels of item features.
    item_labels: np.array of strings of shape [n_items,]
         Items' titles.
    """

    if not (indicator_features or genre_features):
        raise ValueError('At least one of item_indicator_features '
                         'or genre_features must be True')

    zip_path = _common.get_data(data_home,
                                'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                                'movielens100k',
                                'movielens.zip',
                                download_if_missing)

    # Load raw data
    raw = _read_raw_data(zip_path)

    # Figure out the dimensions
    num_users, _, max_timestamp = _get_dimensions(_parse(raw))

    train, test = _build_interaction_matrices(num_users,
                                              max_timestamp,
                                              _parse(raw),
                                              min_rating)
    data = {'train': train.tocsr(),
            'test': test.tocsr()}

    return data
