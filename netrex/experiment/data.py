from lightfm.datasets import fetch_movielens

from lightfm import LightFM


def get_movielens_100k():

    movielens = fetch_movielens()
    train, test = movielens['train'], movielens['test']

    all_ratings = (train + test).tocsr()

    for uid, row in enumerate(all_ratings):
        iids = row
