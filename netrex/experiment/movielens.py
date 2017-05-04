import array
import zipfile

import numpy as np

import scipy.sparse as sp

from netrex.experiment import _common


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

        yield int(uid), int(iid), int(rating), int(timestamp)


def _load_data(data):

    user_dict = {}
    item_dict = {}

    uids = array.array('i')
    iids = array.array('i')
    timestamps = array.array('i')

    for uid, iid, rating, timestamp in _parse(data):
        uids.append(user_dict.setdefault(uid,
                                         len(user_dict)))
        iids.append(item_dict.setdefault(iid,
                                         len(item_dict)))
        timestamps.append(timestamp)

    return (np.array(uids),
            np.array(iids),
            np.array(timestamps))
        

def fetch_movielens(data_home=None, download_if_missing=True, random_seed=None):
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
    download_if_missing: bool, optional
        Download the data if not present. Raises an IOError if False and data is missing.

    Notes
    -----

    The return value is a dictionary containing the following keys:

    Returns
    -------

    """

    random_state = np.random.RandomState(random_seed)

    validation_percentage = 0.1
    test_percentage = 0.1
    
    zip_path = _common.get_data(data_home,
                                ('https://github.com/maciejkula/'
                                 'lightfm_datasets/releases/'
                                 'download/v0.1.0/movielens.zip'),
                                'movielens100k',
                                'movielens.zip',
                                download_if_missing)

    # Load raw data
    raw = _read_raw_data(zip_path)

    uids, iids, timestamps = _load_data(raw)

    num_users = uids.max() + 1
    num_items = iids.max() + 1

    sort_indices = np.arange(len(timestamps))
    random_state.shuffle(sort_indices)

    uids = uids[sort_indices]
    iids = iids[sort_indices]
    timestamps = timestamps[sort_indices]

    validation_index = int(validation_percentage * len(uids))
    test_index = int((validation_percentage + test_percentage) * len(uids))

    validation = sp.coo_matrix((np.ones_like(uids[:validation_index]),
                                (uids[:validation_index],
                                 iids[:validation_index])),
                               shape=(num_users, num_items))
    test = sp.coo_matrix((np.ones_like(uids[validation_index:test_index]),
                                (uids[validation_index:test_index],
                                 iids[validation_index:test_index])),
                               shape=(num_users, num_items))
    train = sp.coo_matrix((np.ones_like(uids[test_index:]),
                           (uids[test_index:],
                            iids[test_index:])),
                          shape=(num_users, num_items))

    return train, test, validation
