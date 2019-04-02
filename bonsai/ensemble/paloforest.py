"""
This class implements PaloForest, an ensemble of PaloBoost
"""

# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

from bonsai.ensemble.paloboost import PaloBoost
import numpy as np


class PaloForest():

    def __init__(self,
                n_paloboost = 10,
                distribution = "gaussian",
                learning_rate = 0.1,
                subsample0 = 0.7,
                subsample1 = 0.7,
                subsample2 = 0.7, 
                max_depth = 3,
                n_estimators = 100, 
                block_size = None,
                random_state = 0):
        self.n_paloboost = n_paloboost
        self.distribution = distribution
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample0 = subsample0 # subsample rate at the forest level
        self.subsample1 = subsample1 # subsample rate at the base level
        self.subsample2 = subsample2 # subsample rate for the columns
        self.block_size = block_size # block sampling size for subsample0
        self.random_state = random_state
        self.estimators = []
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n, m = X.shape
        idx = np.arange(n)
        self.estimators = []
        for i in range(self.n_paloboost):
            mask = np.full(n, True)
            if self.block_size is not None:
                n_block = int(n/self.block_size) + 1
                mask_block = (np.random.rand(n_block) < self.subsample0)
                mask = np.repeat(mask_block, self.block_size)[:n]
            else:
                mask = (np.random.rand(n) < self.subsample0)
            
            X_i, y_i = X[mask,:], y[mask]
            est = PaloBoost(distribution=self.distribution,
                               learning_rate=self.learning_rate,
                                max_depth=self.max_depth,
                                n_estimators=self.n_estimators,
                                subsample=self.subsample1,
                                subsample_splts=self.subsample2,
                                random_state=i*self.n_estimators)
            est.fit(X_i, y_i)
            self.estimators.append(est) 
            if self.feature_importances_ is None:
                self.feature_importances_ = est.feature_importances_
            else:
                self.feature_importances_ += est.feature_importances_
        self.feature_importances_ /= self.n_paloboost

    def predict(self, X):
        y_hat = None
        for est in self.estimators:
            if y_hat is None:
                y_hat = est.predict(X)
            else: 
                y_hat += est.predict(X)
        y_hat /= len(self.estimators)
        return y_hat

    def predict_proba(self, X):
        return self.predict(X)

    def dump(self): 
        return [estimator.dump() for estimator in self.estimators]
   
    def load(self, model):
        self.estimators = []
        for d in model:
            est = PaloBoost()
            est.load(d)
            self.estimators.append(est)





