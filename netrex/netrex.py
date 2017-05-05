import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable, Function

from netrex.layers import ScaledEmbedding, ZeroEmbedding
from netrex.native import get_lib


def _gpu(tensor, gpu=False):

    if gpu:
        return tensor.cuda()
    else:
        return tensor


def _cpu(tensor):

    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def _minibatch(tensor, batch_size):

    for i in range(0, len(tensor), batch_size):
        yield tensor[i:i + batch_size]


def binarize_array(array):

    assert array.shape[1] % 8 == 0

    array = (np.sign(array) > 0.0).astype(np.bool)
    array = np.packbits(array, axis=1)

    return array
    

def binary_dot(x, y):

    x_scale = x.abs().mean(1).detach()
    y_scale = y.abs().mean(1).detach()

    xnor = sign(x) * sign(y)

    return xnor.sum(1) * x_scale * y_scale


class Sign(Function):

    def forward(self, x):

        self.save_for_backward(x)

        return x.sign()

    def backward(self, grad_output):

        x, = self.saved_tensors

        return grad_output * (x.abs() < 1.0).float().abs()


def sign(x):

    return Sign()(x)


class BilinearNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim,
                 xnor=False,
                 sparse=False):

        super().__init__()

        self.xnor = xnor

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim,
                                               sparse=sparse)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                               sparse=sparse)
        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.view(-1, self.embedding_dim)
        item_embedding = item_embedding.view(-1, self.embedding_dim)

        user_bias = self.user_biases(user_ids).view(-1, 1)
        item_bias = self.item_biases(item_ids).view(-1, 1)

        if self.xnor:
            dot = binary_dot(user_embedding, item_embedding)
        else:
            dot = (user_embedding * item_embedding).sum(1)

        return dot + user_bias + item_bias


class FactorizationModel(object):
    """
    A number of classic factorization models, implemented in PyTorch.

    Available loss functions:
    - pointwise logistic
    - BPR: Rendle's personalized Bayesian ranking
    - adaptive: a variant of WARP with adaptive selection of negative samples
    - regression: minimizing the regression loss between true and predicted ratings
    - truncated_regression: truncated regression model, that jointly models
                            the likelihood of a rating being given and the value
                            of the rating itself.

    Performance notes: neural network toolkits do not perform well on sparse tasks
    like recommendations. To achieve acceptable speed, either use the `sparse` option
    on a CPU or use CUDA with very big minibatches (1024+).
    """

    def __init__(self,
                 loss='pointwise',
                 xnor=False,
                 embedding_dim=64,
                 n_iter=3,
                 batch_size=64,
                 l2=0.0,
                 use_cuda=False,
                 sparse=False):

        assert loss in ('pointwise',
                        'bpr',
                        'adaptive')

        self._loss = loss
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._xnor = xnor

        self._num_users = None
        self._num_items = None
        self._net = None

    def _pointwise_loss(self, users, items, ratings):

        negatives = Variable(
            _gpu(
                torch.from_numpy(np.random.randint(0,
                                                   self._num_items,
                                                   len(users))),
                self._use_cuda)
        )

        positives_loss = (1.0 - F.sigmoid(self._net(users, items)))
        negatives_loss = F.sigmoid(self._net(users, negatives))

        return torch.cat([positives_loss, negatives_loss]).mean()

    def _bpr_loss(self, users, items, ratings):

        negatives = Variable(
            _gpu(
                torch.from_numpy(np.random.randint(0,
                                                   self._num_items,
                                                   len(users))),
                self._use_cuda)
        )

        return (1.0 - F.sigmoid(self._net(users, items) -
                                self._net(users, negatives))).mean()

    def _adaptive_loss(self, users, items, ratings,
        n_neg_candidates=5):
        negatives = Variable(
            _gpu(
                torch.from_numpy(
                    np.random.randint(0, self._num_items,
                        (len(users), n_neg_candidates))),
                self._use_cuda)
        )
        negative_predictions = self._net(
            users.repeat(n_neg_candidates, 1).transpose(0,1),
            negatives
            ).view(-1, n_neg_candidates)

        best_negative_prediction, _ = negative_predictions.max(1)
        positive_prediction = self._net(users, items)

        return torch.mean(torch.clamp(best_negative_prediction -
                                      positive_prediction
                                      + 1.0, 0.0))

    def _shuffle(self, interactions):

        users = interactions.row
        items = interactions.col
        ratings = interactions.data

        shuffle_indices = np.arange(len(users))
        np.random.shuffle(shuffle_indices)

        return (users[shuffle_indices].astype(np.int64),
                items[shuffle_indices].astype(np.int64),
                ratings[shuffle_indices].astype(np.float32))

    def fit(self, interactions, verbose=False):
        """
        Fit the model.

        Arguments
        ---------

        interactions: np.float32 coo_matrix of shape [n_users, n_items]
             the matrix containing
             user-item interactions.
        verbose: Bool, optional
             Whether to print epoch loss statistics.
        """

        self._num_users, self._num_items = interactions.shape

        self._net = _gpu(
            BilinearNet(self._num_users,
                        self._num_items,
                        self._embedding_dim,
                        xnor=self._xnor,
                        sparse=self._sparse),
            self._use_cuda
        )

        if self._sparse:
            optimizer = optim.Adagrad(self._net.parameters(),
                                      weight_decay=self._l2)
        else:
            optimizer = optim.Adam(self._net.parameters(),
                                   weight_decay=self._l2)

        if self._loss == 'pointwise':
            loss_fnc = self._pointwise_loss
        elif self._loss == 'bpr':
            loss_fnc = self._bpr_loss
        else:
            loss_fnc = self._adaptive_loss

        for epoch_num in range(self._n_iter):

            users, items, ratings = self._shuffle(interactions)

            user_ids_tensor = _gpu(torch.from_numpy(users),
                                   self._use_cuda)
            item_ids_tensor = _gpu(torch.from_numpy(items),
                                   self._use_cuda)
            ratings_tensor = _gpu(torch.from_numpy(ratings),
                                  self._use_cuda)

            epoch_loss = 0.0

            for (batch_user,
                 batch_item,
                 batch_ratings) in zip(_minibatch(user_ids_tensor,
                                                  self._batch_size),
                                       _minibatch(item_ids_tensor,
                                                  self._batch_size),
                                       _minibatch(ratings_tensor,
                                                  self._batch_size)):

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)
                ratings_var = Variable(batch_ratings)

                optimizer.zero_grad()

                loss = loss_fnc(user_var, item_var, ratings_var)
                epoch_loss += loss.data[0]

                loss.backward()
                # return loss
                optimizer.step()

            if verbose:
                print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    def predict(self, user_ids, item_ids, ratings=False):
        """
        Compute the recommendation score for user-item pairs.

        Arguments
        ---------

        user_ids: integer or np.int32 array of shape [n_pairs,]
             single user id or an array containing the user ids for the user-item pairs for which
             a prediction is to be computed
        item_ids: np.int32 array of shape [n_pairs,]
             an array containing the item ids for the user-item pairs for which
             a prediction is to be computed.
        ratings: bool, optional
             Return predictions on ratings (rather than likelihood of rating)
        """

        if ratings:
            if self._loss not in ('regression',
                                  'truncated_regression'):
                raise ValueError('Ratings can only be returned '
                                 'when the truncated regression loss '
                                 'is used')

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(_gpu(user_ids, self._use_cuda))
        item_var = Variable(_gpu(item_ids, self._use_cuda))

        out = self._net(user_var, item_var)

        if self._loss in ('truncated_regression',):
            if ratings:
                return _cpu((out[1]).data).numpy().flatten()
            else:
                return _cpu((out[0]).data).numpy().flatten()
        else:
            return _cpu(out.data).numpy().flatten()

    def get_scorer(self):

        get_param = lambda l: _cpu([x for x in l.parameters()][0]).data.numpy()

        if self._xnor is False:
            return Scorer(get_param(self._net.user_embeddings),
                          get_param(self._net.user_biases),
                          get_param(self._net.item_embeddings),
                          get_param(self._net.item_biases))
        else:
            return XNORScorer(get_param(self._net.user_embeddings),
                              get_param(self._net.user_biases),
                              get_param(self._net.item_embeddings),
                              get_param(self._net.item_biases))


class Scorer:

    def __init__(self,
                 user_vectors,
                 user_biases,
                 item_vectors,
                 item_biases):

        assert item_vectors.shape[1] % 8 == 0

        self._user_vectors = user_vectors
        self._user_biases = user_biases
        self._item_vectors = item_vectors
        self._item_biases = item_biases

        self._lib = get_lib()

    def predict(self, user_id):

        return self._lib.predict_float_256(
            self._user_vectors[user_id],
            self._item_vectors,
            self._user_biases[user_id],
            self._item_biases)


class XNORScorer:

    def __init__(self,
                 user_vectors,
                 user_biases,
                 item_vectors,
                 item_biases):

        assert item_vectors.shape[1] % 8 == 0

        self._user_norms = np.abs(user_vectors).mean(axis=1)
        self._item_norms = np.abs(item_vectors).mean(axis=1)

        self._user_vectors = binarize_array(user_vectors)
        self._user_biases = user_biases
        self._item_vectors = binarize_array(item_vectors)
        self._item_biases = item_biases

        self._lib = get_lib()

    def predict(self, user_id):

        return self._lib.predict_xnor_256(
            self._user_vectors[user_id],
            self._item_vectors,
            self._user_biases[user_id],
            self._item_biases,
            self._user_norms[user_id],
            self._item_norms)
