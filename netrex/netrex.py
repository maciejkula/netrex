import numpy as np

import scipy.sparse as sp
import scipy.stats as st

from sklearn.metrics import roc_auc_score

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

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


class PointwiseSampleNet(nn.Module):

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


class PairwiseSampleNet(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_embeddings = ScaledEmbedding(num_users, embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_items, embedding_dim)
        self.item_biases = ZeroEmbedding(num_items, 1)

    def forward(self, user_ids, item_ids):

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.view(-1, self.embedding_dim)
        item_embedding = item_embedding.view(-1, self.embedding_dim)

        item_bias = self.item_biases(item_ids).view(-1, 1)

        sigm_in = (user_embedding * item_embedding).sum(1)

        return sigm_in + item_bias


# class PairwiseSampleNet(nn.Module):

#     def __init__(self, num_users, num_items, embedding_dim):
#         super().__init__()

#         self.embedding_dim = embedding_dim

#         self.user_embeddings = nn.Embedding(num_users, embedding_dim)
#         self.item_embeddings = nn.Embedding(num_items, embedding_dim)

#         self.fc1 = nn.Linear(embedding_dim * 2, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 1)

#     def forward(self, user_ids, item_ids):

#         user_embedding = self.user_embeddings(user_ids)
#         item_embedding = self.item_embeddings(item_ids)

#         user_embedding = user_embedding.view(-1, self.embedding_dim)
#         item_embedding = item_embedding.view(-1, self.embedding_dim)

#         x = torch.cat([user_embedding, item_embedding], 1)
#         x = F.sigmoid(self.fc1(x))
#         x = F.sigmoid(self.fc2(x))
#         x = self.fc3(x)

#         return x


def sample_negatives(user_ids, item_ids):

    user_ids = np.concatenate([user_ids, user_ids])

    y = np.ones(len(item_ids) * 2, dtype=np.float32)
    y[len(item_ids):] = 0.0

    item_ids = np.concatenate([item_ids,
                               np.random.choice(item_ids, len(item_ids))])

    shuffle_indices = np.arange(len(user_ids))
    np.random.shuffle(shuffle_indices)

    return (user_ids[shuffle_indices].astype(np.int64),
            item_ids[shuffle_indices].astype(np.int64),
            y[shuffle_indices].astype(np.float32))


def sample_triplets(user_ids, item_ids):

    negative_item_ids = np.random.choice(item_ids, len(item_ids))
    # negative_item_ids = np.random.randint(0, item_ids.max() + 1, len(item_ids))

    shuffle_indices = np.arange(len(user_ids))
    np.random.shuffle(shuffle_indices)

    return (user_ids[shuffle_indices].astype(np.int64),
            item_ids[shuffle_indices].astype(np.int64),
            negative_item_ids[shuffle_indices].astype(np.int64))


class PointwiseImplicitFactorization(object):

    def __init__(self,
                 embedding_dim=64,
                 n_iter=3,
                 batch_size=64,
                 use_cuda=False):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._use_cuda = use_cuda

        self._num_users = None
        self._num_items = None
        self._net = None

    def fit(self, interactions):

        self._num_users, self._num_items = interactions.shape

        self._net = _gpu(PointwiseSampleNet(self._num_users,
                                            self._num_items,
                                            self._embedding_dim), self._use_cuda)

        criterion = nn.BCELoss()
        # optimizer = optim.Adagrad(self._net.parameters(), lr=0.5)
        optimizer = optim.Adam(self._net.parameters())

        for epoch_num in range(self._n_iter):

            user_ids, item_ids, y = sample_negatives(interactions.row,
                                                     interactions.col)

            user_ids_tensor = _gpu(torch.from_numpy(user_ids), self._use_cuda)
            item_ids_tensor = _gpu(torch.from_numpy(item_ids), self._use_cuda)
            y_tensor = _gpu(torch.from_numpy(y), self._use_cuda)

            for (batch_user, batch_item, batch_y) in zip(_minibatch(user_ids_tensor, self._batch_size),
                                                         _minibatch(item_ids_tensor, self._batch_size),
                                                         _minibatch(y_tensor, self._batch_size)):

                user_var = Variable(batch_user)
                item_var = Variable(batch_item)
                y_var = Variable(batch_y)

                optimizer.zero_grad()

                y_hat = F.sigmoid(self._net.forward(user_var, item_var))
                loss = criterion(y_hat, y_var)

                # if not np.isfinite(_cpu(loss).data.numpy()).all():
                #     assert False

                loss.backward()
                optimizer.step()

            print('Epoch {}: loss {}'.format(epoch_num, loss.data[0]))
            try:
                print('AUC: {}'.format(roc_auc_score(y, self.predict(user_ids, item_ids))))
            except Exception as e:
                print(e)
                raise e

    def predict(self, user_ids, item_ids):

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(_gpu(user_ids, self._use_cuda))
        item_var = Variable(_gpu(item_ids, self._use_cuda))

        return _cpu(self._net(user_var, item_var).data).numpy().flatten()


class PairwiseImplicitFactorization(object):

    def __init__(self,
                 embedding_dim=64,
                 n_iter=3,
                 batch_size=64,
                 use_cuda=False):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._use_cuda = use_cuda

        self._num_users = None
        self._num_items = None
        self._net = None

    def fit(self, interactions):

        self._num_users, self._num_items = interactions.shape

        self._net = _gpu(PointwiseSampleNet(self._num_users,
                                            self._num_items,
                                            self._embedding_dim),
                         self._use_cuda)

        # criterion = nn.BCELoss()
        #optimizer = optim.Adagrad(self._net.parameters(), lr=0.01)
        optimizer = optim.Adam(self._net.parameters(), weight_decay=0.0001)

        for epoch_num in range(self._n_iter):

            (user_ids,
             item_ids,
             negative_item_ids) = sample_triplets(interactions.row,
                                                  interactions.col)

            user_ids_tensor = _gpu(torch.from_numpy(user_ids), self._use_cuda)
            item_ids_tensor = _gpu(torch.from_numpy(item_ids), self._use_cuda)
            negative_item_ids_tensor = _gpu(torch.from_numpy(negative_item_ids), self._use_cuda)

            epoch_loss = 0.0

            for (batch_user, batch_pos, batch_neg) in zip(_minibatch(user_ids_tensor, self._batch_size),
                                                          _minibatch(item_ids_tensor, self._batch_size),
                                                          _minibatch(negative_item_ids_tensor, self._batch_size)):

                user_var = Variable(batch_user)
                pos_var = Variable(batch_pos)
                neg_var = Variable(batch_neg)

                optimizer.zero_grad()

                pos_pred = self._net(user_var, pos_var)
                neg_pred = self._net(user_var, neg_var)

                loss = torch.mean(1.0 - F.sigmoid(pos_pred - neg_pred))

                epoch_loss += loss.data[0]
                loss.backward()
                optimizer.step()

            print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    def predict(self, user_ids, item_ids):

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(_gpu(user_ids, self._use_cuda))
        item_var = Variable(_gpu(item_ids, self._use_cuda))

        return _cpu(F.sigmoid(self._net(user_var, item_var)).data).numpy().flatten()


class ActivePairwiseImplicitFactorization(object):

    def __init__(self,
                 embedding_dim=64,
                 n_iter=3,
                 batch_size=64,
                 use_cuda=False):

        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._batch_size = batch_size
        self._use_cuda = use_cuda

        self._num_users = None
        self._num_items = None
        self._net = None

    def fit(self, interactions):

        self._num_users, self._num_items = interactions.shape

        self._net = _gpu(PointwiseSampleNet(self._num_users,
                                            self._num_items,
                                            self._embedding_dim),
                         self._use_cuda)

        # criterion = nn.BCELoss()
        #optimizer = optim.Adagrad(self._net.parameters(), lr=0.01)
        optimizer = optim.Adam(self._net.parameters(), weight_decay=0.0001)

        for epoch_num in range(self._n_iter):

            (user_ids,
             item_ids,
             negative_item_ids) = sample_triplets(interactions.row,
                                                  interactions.col)

            user_ids_tensor = _gpu(torch.from_numpy(user_ids), self._use_cuda)
            item_ids_tensor = _gpu(torch.from_numpy(item_ids), self._use_cuda)
            negative_item_ids_tensor = _gpu(torch.from_numpy(negative_item_ids), self._use_cuda)

            epoch_loss = 0.0

            for (batch_user, batch_pos, batch_neg) in zip(_minibatch(user_ids_tensor, self._batch_size),
                                                          _minibatch(item_ids_tensor, self._batch_size),
                                                          _minibatch(negative_item_ids_tensor, self._batch_size)):

                user_var = Variable(batch_user)
                pos_var = Variable(batch_pos)
                neg_var = Variable(batch_neg)

                # Multiple negative passes
                neg_preds = []
                
                for _ in range(5):
                    neg_var = Variable(_gpu(
                        torch.from_numpy(np.random.randint(0, self._num_items, len(batch_user))),
                        self._use_cuda)
                    )
                    neg_preds.append(self._net(user_var, neg_var))

                neg_pred, _ = torch.cat(neg_preds, 1).max(1)

                optimizer.zero_grad()

                pos_pred = self._net(user_var, pos_var)
                # neg_pred = self._net(user_var, neg_var)

                # loss = torch.mean(1.0 - F.sigmoid(pos_pred - neg_pred))
                loss = torch.mean(torch.clamp(neg_pred - pos_pred + 1.0, 0.0))

                epoch_loss += loss.data[0]
                loss.backward()
                optimizer.step()

            print('Epoch {}: loss {}'.format(epoch_num, epoch_loss))

    def predict(self, user_ids, item_ids):

        user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
        item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

        user_var = Variable(_gpu(user_ids, self._use_cuda))
        item_var = Variable(_gpu(item_ids, self._use_cuda))

        return _cpu(F.sigmoid(self._net(user_var, item_var)).data).numpy().flatten()
