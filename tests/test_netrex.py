import argparse

import numpy as np

from netrex.netrex import ImplicitFactorizationModel
from netrex.evaluation import auc_score, mrr_score

from lightfm.datasets import fetch_movielens

from lightfm import LightFM
import lightfm.evaluation


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--sparse', action='store_true')

    args = parser.parse_args()
    cuda = args.gpu
    sparse = args.sparse

    def _binarize(dataset):

        dataset = dataset.copy()

        dataset.data = (dataset.data >= 0.0).astype(np.float32)
        dataset = dataset.tocsr()
        dataset.eliminate_zeros()

        return dataset.tocoo()

    movielens = fetch_movielens()
    ratings_train, ratings_test = movielens['train'], movielens['test']
    train, test = _binarize(movielens['train']), _binarize(movielens['test'])

    embedding_dim = 128

    # lfm = LightFM(no_components=embedding_dim, loss='warp')
    # lfm.fit(train, epochs=5)
    # print(auc_score(lfm, test, train).mean())
    # print(mrr_score(lfm, test, train).mean())

    l2 = 0.0

    for loss in ('regression', 'truncated_regression'):
        print('Model loss: {}'.format(loss))

        model = ImplicitFactorizationModel(loss=loss,
                                           n_iter=20,
                                           l2=l2,
                                           embedding_dim=embedding_dim,
                                           use_cuda=cuda,
                                           sparse=sparse)

        model.fit(ratings_train)

        print(auc_score(model, test, train).mean())
        print(mrr_score(model, test, train).mean())
        print('RMSE:')
        print(np.sqrt(((model.predict(ratings_test.row, ratings_test.col, ratings=True)
                        - ratings_test.data) ** 2).mean()))
        print('RMSE training set:')
        print(np.sqrt(((model.predict(ratings_train.row, ratings_train.col, ratings=True)
                        - ratings_train.data) ** 2).mean()))

        print(model.predict(ratings_test.row, ratings_test.col, ratings=True))

    embedding_dim *= 2

    for loss in ('pointwise', 'bpr', 'adaptive'):
        print('Model loss: {}'.format(loss))

        model = ImplicitFactorizationModel(loss=loss,
                                           l2=l2,
                                           n_iter=5,
                                           embedding_dim=embedding_dim,
                                           use_cuda=cuda,
                                           sparse=sparse)

        model.fit(train)

        print(auc_score(model, test, train).mean())
        print(mrr_score(model, test, train).mean())
