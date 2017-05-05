import numpy as np

from netrex.netrex import FactorizationModel
from netrex.native import get_lib
from netrex.experiment import movielens


def _predict_float_256(user_vector,
                       item_vectors,
                       user_bias,
                       item_biases):

    return user_bias + item_biases + np.dot(item_vectors, user_vector)


def _predict_xnor_256(xnor_scorer, user_id):

    biases = xnor_scorer._user_biases[user_id] + xnor_scorer._item_biases

    user_vectors = np.tile(np.unpackbits(xnor_scorer._user_vectors[user_id]),
                           (len(biases), 1))
    item_vectors = np.unpackbits(xnor_scorer._item_vectors, axis=1)

    user_norm = xnor_scorer._user_norms[user_id]
    item_norms = xnor_scorer._item_norms

    matching = (user_vectors == item_vectors).sum(axis=1)
    not_matching = (user_vectors != item_vectors).sum(axis=1)

    return (matching - not_matching) * item_norms * user_norm + biases


def _get_model(xnor=False, embedding_dim=256):

    train, test, validation = movielens.fetch_movielens(random_seed=402)

    n_iter = 0
    minibatch_size = 4096
    loss = 'bpr'
    l2 = 0.0

    model = FactorizationModel(loss=loss,
                               embedding_dim=embedding_dim,
                               use_cuda=False,
                               xnor=xnor,
                               n_iter=n_iter)
    model.fit(train)

    return model
                       


def test_predict_float_256():

    lib = get_lib()

    latent_dim = 128
    num_items = 1024

    user_vector = np.random.random(latent_dim).astype(np.float32)
    item_vectors = np.random.random((num_items, latent_dim)).astype(np.float32)
    user_bias = 1.0
    item_biases = np.random.random(num_items).astype(np.float32)

    expected = _predict_float_256(user_vector,
                                  item_vectors,
                                  user_bias,
                                  item_biases)
    ffi_result = lib.predict_float_256(user_vector,
                                       item_vectors,
                                       user_bias,
                                       item_biases)

    assert np.allclose(expected, ffi_result)


def test_scorer():

    model = _get_model()
    scorer = model.get_scorer()

    item_ids = np.arange(len(scorer._item_biases))

    expected = model.predict(np.repeat(0, len(item_ids)), item_ids)
    predictions = scorer.predict(0)

    assert np.allclose(expected, predictions, atol=0.000001)


def test_xnor_scorer():

    model = _get_model(xnor=True)
    scorer = model.get_scorer()

    item_ids = np.arange(len(scorer._item_biases))

    expected = _predict_xnor_256(scorer, 0)
    predictions = scorer.predict(0)

    assert np.allclose(expected, predictions, atol=0.000001)

    expected = model.predict(np.repeat(0, len(item_ids)), item_ids)
    assert np.allclose(expected, predictions, atol=0.000001)



model = _get_model(xnor=True, embedding_dim=2048)
scorer = model.get_scorer()

model = _get_model(xnor=False, embedding_dim=2048)
scorer = model.get_scorer()
