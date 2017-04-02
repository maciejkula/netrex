import os

from dynarray import DynamicArray

import numpy as np

import requests

import scipy.sparse as sp


def _get_data_dir():

    return os.path.join(os.path.expanduser('~'),
                        '_netrex_data')


def _create_data_dir(path):

    if not os.path.isdir(path):
        os.makedirs(path)


def _download(url, dest_path):

    req = requests.get(url, stream=True)

    with open(dest_path, 'wb') as fd:
        for chunk in req.iter_content(chunk_size=2**20):
            fd.write(chunk)


def _download_if_not_exists(url, object_name):

    _create_data_dir(_get_data_dir())

    path = os.path.join(_get_data_dir(),
                        object_name)

    if not os.path.exists(path):
        _download(url, path)

    return path


def _read_amazon_ratings(subset):

    object_name = 'ratings_{}.csv'.format(subset.title())

    url = ('http://snap.stanford.edu/data/amazon/'
           'productGraph/categoryFiles/{}').format(object_name)

    path = _download_if_not_exists(url, object_name)

    user_dict = {}
    item_dict = {}

    with open(path) as data:
        for line in data:
            user, item, rating, timestamp = line.replace('\n', '').split(',')

            user = user_dict.setdefault(user, len(user_dict))
            item = item_dict.setdefault(item, len(item_dict))
            rating = float(rating)
            timestamp = int(timestamp)

            yield user, item, rating, timestamp


def get_amazon_ratings(subset, test_set_fraction=0.1, test_user_fraction=0.1, min_ratings=5):

    users = DynamicArray(dtype=np.int32)
    items = DynamicArray(dtype=np.int32)
    ratings = DynamicArray(dtype=np.float32)
    timestamps = DynamicArray(dtype=np.int32)

    for user, item, rating, timestamp in _read_amazon_ratings(subset):
        users.append(user)
        items.append(item)
        ratings.append(rating)
        timestamps.append(timestamp)

    shape = users[:].max() + 1, items[:].max() + 1

    test_cutoff_index = int(len(timestamps) * (1.0 - test_set_fraction))
    test_cutoff_timestamp = np.sort(timestamps[:])[test_cutoff_index]
    in_train = timestamps < test_cutoff_timestamp
    in_test_time = np.logical_not(in_train)

    unique_profiles_after_split = np.unique(users[in_test_time])
    test_and_validation = np.random.choice(unique_profiles_after_split,
                                           int(test_user_fraction
                                               * len(unique_profiles_after_split)))
    test_users, validation_users = np.array_split(test_and_validation, 2)
    test_users, validation_users = set(test_users), set(validation_users)

    in_test = np.logical_and(in_test_time, np.array([x in test_users for x in users]))
    in_validation = np.logical_and(in_test_time, np.array([x in validation_users for x in users]))

    return (
        sp.coo_matrix((ratings[in_train],
                       (users[in_train],
                        items[in_train])),
                      shape=shape),
        sp.coo_matrix((ratings[in_test],
                       (users[in_test],
                        items[in_test])),
                      shape=shape),
        sp.coo_matrix((ratings[in_validation],
                       (users[in_validation],
                        items[in_validation])),
                      shape=shape),
    )
