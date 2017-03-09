import numpy as np

from netrex.netrex import ImplicitFactorizationModel
from netrex.evaluation import auc_score, mrr_score

from lightfm.datasets import fetch_movielens

from lightfm import LightFM
import lightfm.evaluation

if __name__ == '__main__':

    def _binarize(dataset):

        dataset.data = (dataset.data >= 3.0).astype(np.float32)
        dataset = dataset.tocsr()
        dataset.eliminate_zeros()

        return dataset.tocoo()

    movielens = fetch_movielens()
    train, test = _binarize(movielens['train']), _binarize(movielens['test'])

    embedding_dim = 32

    # lfm = LightFM(no_components=embedding_dim, loss='warp')
    # lfm.fit(train, epochs=5)
    # print(auc_score(lfm, test, train).mean())
    # print(mrr_score(lfm, test, train).mean())

    for loss in ('pointwise', 'bpr', 'adaptive'):
        model = ImplicitFactorizationModel(loss=loss,
                                           n_iter=5,
                                           embedding_dim=embedding_dim,
                                           use_cuda=False)

        model.fit(train)

        print(auc_score(model, test, train).mean())
        print(mrr_score(model, test, train).mean())
