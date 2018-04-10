import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

import pyhdi

class Logger(pyhdi.HDILogger):
    def __init__(self):
        pyhdi.HDILogger.__init__(self)

    def log(self, level, msg):
        print(level, msg)

class HighDimensionalInspectorTSNE(BaseEstimator):
    """t-distributed Stochastic Neighbor Embedding.
    """

    def __init__(self, n_components=2, method="asne", n_iter=1000, perplexity=30.0, seed=0, minimum_gain=0.1, eta=200.,
                 momentum=0.5, final_momentum=0.8, mom_switching_iter=250, exaggeration_factor=4.0,
                 remove_exaggeration_iter=250, theta=0.5):
        self.n_components = n_components
        self.method = method
        self.n_iter = n_iter
        self.perplexity = perplexity
        self.seed = seed
        self.minimum_gain = minimum_gain
        self.eta = eta
        self.momentum = momentum
        self.final_momentum = final_momentum
        self.mom_switching_iter = mom_switching_iter
        self.exaggeration_factor = exaggeration_factor
        self.remove_exaggeration_iter = remove_exaggeration_iter
        self.theta = theta

    def _check_parameters(self):
        if self.method not in ["asne", "tsne"]:
            raise ValueError("'method' must be either 'tsne' or 'asne'")

        if not 0.0 < self.minimum_gain < 1.0:
            raise ValueError("'minimum_gain' must be between 0.0 and 1.0")

        if self.exaggeration_factor < 1.0:
            raise ValueError("exaggeration_factor must be at least 1, but is {}".format(self.exaggeration_factor))

        if self.n_iter < 250:
            raise ValueError("n_iter should be at least 250")

    def _fit(self, X):
        """Fit the model using X as training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        """
        X = check_array(X, accept_sparse=[], dtype=[np.double])

        self._check_parameters()

        n_points = X.shape[0]
        embedding = np.asarray(np.zeros((n_points, self.n_components)), dtype=np.double)

        tsne = pyhdi.HDItSNE() if self.method == "tsne" else pyhdi.HDIaSNE()
        tsne.parameters().set_logger(Logger().__disown__())
        tsne.parameters().set_n_points(n_points)
        tsne.parameters().set_input(X)
        tsne.parameters().set_output(embedding)
        tsne.parameters().set_perplexity(self.perplexity)
        tsne.parameters().set_seed(self.seed)
        tsne.parameters().set_minimum_gain(self.minimum_gain)
        tsne.parameters().set_eta(self.eta)
        tsne.parameters().set_momentum(self.momentum)
        tsne.parameters().set_final_momentum(self.final_momentum)
        tsne.parameters().set_mom_switching_iter(self.mom_switching_iter)
        tsne.parameters().set_exaggeration_factor(self.exaggeration_factor)
        tsne.parameters().set_remove_exaggeration_iter(self.remove_exaggeration_iter)
        tsne.parameters().set_theta(self.theta)
        tsne.run(self.n_iter)
        tsne.parameters().del_logger()

        return embedding

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        y : Ignored.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        y : Ignored.
        """
        self.fit_transform(X)
        return self
