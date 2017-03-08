import numpy as np

import scipy.stats as st

from sklearn.metrics import roc_auc_score


def mrr_score(model, test, train=None):

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    item_ids = np.arange(test.shape[1])
    user_ids = np.empty(test.shape[1])

    mrrs = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        user_ids.fill(user_id)

        predictions = -model.predict(user_ids, item_ids)

        if train is not None:
            predictions[train[user_id].indices] = np.finfo(np.float32).max

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


def auc_score(model, test, train=None):

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    item_ids = np.arange(test.shape[1])
    user_ids = np.empty(test.shape[1])

    aucs = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        user_ids.fill(user_id)

        predictions = model.predict(user_ids, item_ids)

        if train is not None:
            predictions[train[user_id].indices] = np.finfo(np.float32).max

        auc = roc_auc_score(np.squeeze(np.array(row.todense())), predictions)

        aucs.append(auc)

    return np.array(aucs)
