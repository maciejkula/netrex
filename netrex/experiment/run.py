import argparse

import numpy as np
import scipy.sparse as sp

from netrex.netrex import FactorizationModel
from netrex.evaluation import auc_score, mrr_score

from sklearn.model_selection import ParameterSampler
from scipy.stats import distributions


def optimize(train, test, xnor, loss='bpr',
             iterations=10,
             minibatch_size=4096,
             random_state=None):

    space = {
        'embedding_dim': [32, 64, 128, 256, 512, 1024, 2048],
        'n_iter': distributions.randint(5, 50),
    }

    sampler = ParameterSampler(space,
                               n_iter=iterations,
                               random_state=random_state)

    results = []

    for hyperparam_set in sampler:
        print(hyperparam_set)
        model = FactorizationModel(loss=loss,
                                   xnor=xnor,
                                   batch_size=minibatch_size,
                                   use_cuda=True,
                                   **hyperparam_set)
        model.fit(train)
        mrr = mrr_score(model, test, train).mean()
        print(mrr)

        results.append((hyperparam_set, mrr))

    return sorted(results, key=lambda x: -x[1])[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--xnor', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--sparse', action='store_true')

    args = parser.parse_args()
    cuda = args.gpu
    sparse = args.sparse
    xnor = args.xnor

    from netrex.experiment import movielens
    train, test, validation = movielens.fetch_movielens(random_seed=402)

    embedding_dim = 256
    n_iter = 20
    minibatch_size = 4096
    loss = 'bpr'
    l2 = 0.0

    print('Model loss: {}'.format(loss))
    hyperparams, _ = optimize(train, test, xnor, iterations=10,
                              minibatch_size=minibatch_size,
                              loss=loss, random_state=402)

    model = FactorizationModel(loss=loss,
                               xnor=xnor,
                               use_cuda=True,
                               sparse=False,
                               batch_size=minibatch_size,
                               **hyperparams)

    loss = model.fit(train, verbose=True)

    print(auc_score(model, validation, train + test).mean())
    print(mrr_score(model, validation, train + test).mean())
