import numpy as np

from netrex.netrex import Scorer, XNORScorer

num_users, num_items, latent_dim = 1, 100000, 1024

user_vectors = np.random.random((num_users, latent_dim)).astype(np.float32)
item_vectors = np.random.random((num_items, latent_dim)).astype(np.float32)
user_biases = np.random.random(num_users).astype(np.float32)
item_biases = np.random.random(num_items).astype(np.float32)

args = (user_vectors, user_biases, item_vectors, item_biases)

scorer = Scorer(*args)
xnor_scorer = XNORScorer(*args)

run = lambda: scorer.predict(0)
run_xnor = lambda: xnor_scorer.predict(0)
