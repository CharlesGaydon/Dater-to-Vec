from functools import partial
import numpy as np
from scipy.linalg import sqrtm
from scipy.sparse.linalg import svds


class FactorizationRecommender:
    def __init__(self, k):
        self.k = k

        self.U = None
        self.s_root = None
        self.V = None
        self.mu_hat = None

        self.rater_index_dict = None
        self.rated_index_dict = None

    # Modified from https://towardsdatascience.com/beginners-guide-to-creating-an-svd-recommender-system-1fd7326d1f65
    # This is not ideal since the traingn
    def fit(self, utility_matrix, rater_index, rated_index):

        # save for use in prediction

        self.rater_index_dict = rater_index
        self.rated_index_dict = rated_index

        k = self.k

        # The magic happens here. U and V are user and item features
        U, s, Vt, mu_hat = self.emsvd(utility_matrix, max_iter=50)

        # we take only the k most significant features
        self.U = U[:, 0:k]
        self.Vt = Vt[0:k, :]
        s = np.diag(s)
        self.s = s
        s = s[0:k, 0:k]
        self.s_root = sqrtm(s)
        # and keep the mean ratings of each raters that were removed
        self.mu_hat = mu_hat

        return self.U, self.s_root, self.Vt, self.mu_hat

    def emsvd(self, utility_matrix, tol=1e-3, max_iter=None):
        """
        Approximate SVD on data with missing values via expectation-maximization

        Inputs:
        -----------
        Y:          (nobs, ndim) data matrix, missing values denoted by NaN/Inf
        tol:        convergence tolerance on change in trace norm
        maxiter:    maximum number of EM steps to perform (default: no limit)

        Returns:
        -----------
        Y_hat:      (nobs, ndim) reconstructed data matrix
        mu_hat:     (ndim,) estimated column means for reconstructed data
        U, s, Vt:   singular values and vectors (see np.linalg.svd and
                    scipy.sparse.linalg.svds for details)
        """
        k = self.k
        svdmethod = partial(svds, k=k)
        if max_iter is None:
            max_iter = np.inf

        # initialize the missing values to their respective column means
        mu_hat = np.nanmean(utility_matrix, axis=0, keepdims=1)

        valid = np.isfinite(utility_matrix)
        Y_hat = np.where(valid, utility_matrix, mu_hat)

        halt = False
        ii = 1
        v_prev = 0

        while not halt:
            print(ii)
            # SVD on filled-in data
            U, s, Vt = svdmethod(Y_hat - mu_hat)

            # impute missing values
            Y_hat[~valid] = (U.dot(np.diag(s)).dot(Vt) + mu_hat)[~valid]

            # update bias parameter
            mu_hat = Y_hat.mean(axis=0, keepdims=1)
            print(mu_hat.shape)
            # test convergence using relative change in trace norm
            v = s.sum()
            if ii >= max_iter or ((v - v_prev) / v_prev) < tol:
                halt = True
            ii += 1
            v_prev = v

        return U, s, Vt, mu_hat

    # taken from https://stackoverflow.com/a/35611142/8086033
    def predict(self, rater_id, rated_id):
        # go from id to index
        rater_idx = self.rater_index_dict[rater_id]
        rated_idx = self.rated_index_dict[rated_id]
        # reconstruct the score from decomposed matrix
        u_s_root = np.dot(self.U[rater_idx, :], self.s_root)  # (k,) array
        s_root_v = np.dot(self.Vt[:, rated_idx], self.s_root)  # (k,1)
        score = np.dot(u_s_root, s_root_v) + self.mu_hat[0, rated_idx]
        return score

    def evaluate(self, test_data, type_of_value="r"):

        if type_of_value == "r":
            rmse = 0
            for row in test_data:
                pred = self.predict(row[0], row[1])
                obs = row[type_of_value]
                rmse += (pred - obs) ** 2
            return rmse / len(test_data)

        if type_of_value == "m":
            pass

    def save(self, path):
        pass

    def load(self, path):
        pass
