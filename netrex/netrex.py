import numpy as np

import scipy.sparse as sp
import scipy.stats as st

from sklearn.metrics import roc_auc_score

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

from netrex.layers import ScaledEmbedding, ZeroEmbedding


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


class BilinearNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)
        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.view(-1, self.embedding_dim)
        item_embedding = item_embedding.view(-1, self.embedding_dim)

        user_bias = self.user_biases(user_ids).view(-1, 1)
        item_bias = self.item_biases(item_ids).view(-1, 1)

        sigm_in = (user_embedding * item_embedding).sum(1)

        return sigm_in + user_bias + item_bias


class TruncatedBilinearNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.rating_net = BilinearNet(num_users, num_items, embedding_dim)
        self.observed_net = BilinearNet(num_users, num_items, embedding_dim)

        self.stddev = nn.Embedding(1, 1)

    def forward(self, user_ids, item_ids):

        observed = F.sigmoid(self.observed_net(user_ids, item_ids))
        rating = self.rating_net(user_ids, item_ids)
        stddev = self.stddev((user_ids < -1).long()).view(-1, 1)

        return observed, rating, stddev


class MultilayerNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)
        self.user_biases = ZeroEmbedding(num_users, 1)
        self.item_biases = ZeroEmbedding(num_items, 1)

        self.fc1 = nn.Linear(embedding_dim * 2, 1)
        # self.fc2 = nn.Linear(embedding_dim * 4, embedding_dim)
        # self.fc3 = nn.Linear(embedding_dim, 1)

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.view(-1, self.embedding_dim)
        item_embedding = item_embedding.view(-1, self.embedding_dim)

        user_bias = self.user_biases(user_ids).view(-1, 1)
        item_bias = self.item_biases(item_ids).view(-1, 1)

        x = torch.cat([user_embedding, item_embedding], 1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.fc1(x)

        return x + user_bias + item_bias


class ImplicitFactorizationModel(object):

    def __init__(self,
                 loss='pointwise',
                 embedding_dim=64,
                 n_iter=3,
                 batch_size=64,
                 l2=0.0,
                 use_cuda=False):

        assert loss in ('pointwise', 'bpr',
                        'adaptive',
                        'truncated_regression',
                        'adaptive_truncated_regression')

        self._loss = loss
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda

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

    def _adaptive_loss(self, users, items, ratings):

        negative_predictions = []

        for _ in range(5):
            negatives = Variable(
                _gpu(
                    torch.from_numpy(np.random.randint(0,
                                                       self._num_items,
                                                       len(users))),
                    self._use_cuda)
            )

            negative_predictions.append(self._net(users, negatives))

        best_negative_prediction, _ = torch.cat(negative_predictions, 1).max(1)
        positive_prediction = self._net(users, items)

        return torch.mean(torch.clamp(best_negative_prediction -
                                      positive_prediction
                                      + 1.0, 0.0))

    def _adaptive_truncated_regression_loss(self, users, items, ratings):

        negative_predictions = []

        for _ in range(5):
            negatives = Variable(
                _gpu(
                    torch.from_numpy(np.random.randint(0,
                                                       self._num_items,
                                                       len(users))),
                    self._use_cuda)
            )

            negative_predictions.append(self._net(users, negatives)[0])

        neg_prob, _ = torch.cat(negative_predictions, 1).max(1)

        pos_prob, pos_rating, pos_stddev = self._net(users, items)

        positives_likelihood = (torch.log(pos_prob)
                                - 0.5 * np.log(2 * np.pi)
                                - 0.5 * torch.log(pos_stddev ** 2)
                                - (0.5 * (pos_rating - ratings) ** 2 / (pos_stddev ** 2)))
        negatives_likelihood = torch.log(1.0 - neg_prob)

        return torch.cat([-positives_likelihood, -negatives_likelihood]).mean()

    def _truncated_regression_loss(self, users, items, ratings):

        negatives = Variable(
            _gpu(
                torch.from_numpy(np.random.randint(0,
                                                   self._num_items,
                                                   len(users))),
                self._use_cuda)
        )

        pos_prob, pos_rating, pos_stddev = self._net(users, items)

        positives_likelihood = (torch.log(pos_prob)
                                - 0.5 * np.log(2 * np.pi)
                                - 0.5 * torch.log(pos_stddev ** 2)
                                - (0.5 * (pos_rating - ratings) ** 2 / (pos_stddev ** 2)))
        neg_prob, _, _ = self._net(users, negatives)
        negatives_likelihood = torch.log(1.0 - neg_prob)

        return torch.cat([-positives_likelihood, -negatives_likelihood]).mean()

    def _shuffle(self, interactions):

        users = interactions.row
        items = interactions.col
        ratings = interactions.data

        shuffle_indices = np.arange(len(users))
        np.random.shuffle(shuffle_indices)

        return (users[shuffle_indices].astype(np.int64),
                items[shuffle_indices].astype(np.int64),
                ratings[shuffle_indices].astype(np.float32))

    def fit(self, interactions):

        self._num_users, self._num_items = interactions.shape

        if self._loss in ('truncated_regression', 'adaptive_truncated_regression'):
            self._net = _gpu(
                TruncatedBilinearNet(self._num_users,
                                     self._num_items,
                                     self._embedding_dim),
                self._use_cuda
            )
        else:
            self._net = _gpu(
                BilinearNet(self._num_users,
                            self._num_items,
                            self._embedding_dim),
                self._use_cuda
            )

        # self._net = _gpu(
        #     MultilayerNet(self._num_users,
        #                   self._num_items,
        #                   self._embedding_dim),
        #     self._use_cuda
        # )

        optimizer = optim.Adam(self._net.parameters(), weight_decay=self._l2)

        if self._loss == 'pointwise':
            loss_fnc = self._pointwise_loss
        elif self._loss == 'bpr':
            loss_fnc = self._bpr_loss
        elif self._loss == 'truncated_regression':
            loss_fnc = self._truncated_regression_loss
        elif self._loss == 'adaptive_truncated_regression':
            loss_fnc = self._adaptive_truncated_regression_loss
        else:
            loss_fnc = self._adaptive_loss

        users, items, ratings = self._shuffle(interactions)

        for epoch_num in range(self._n_iter):

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
                optimizer.step()

            print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    def predict(self, user_ids, item_ids, ratings=False):

        if ratings:
            if self._loss not in ('truncated_regression', 'adaptive_truncated_regression'):
                raise ValueError('Ratings can only be returned '
                                 'when the truncated regression loss '
                                 'is used')

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(_gpu(user_ids, self._use_cuda))
        item_var = Variable(_gpu(item_ids, self._use_cuda))

        out = self._net(user_var, item_var)

        if self._loss in ('truncated_regression', 'adaptive_truncated_regression'):
            if ratings:
                return _cpu((out[1]).data).numpy().flatten()
            else:
                return _cpu((out[0]).data).numpy().flatten()
        else:
            return _cpu(out.data).numpy().flatten()
