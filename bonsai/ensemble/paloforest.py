"""
This class implements PaloBoost, an imprvoed Stochastic Gradient TreeBoost
 that is robust to overfitting and can provide robust performance.
"""

# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

from bonsai.ensemble.paloboost import PaloBoost
import numpy as np


class PaloForest():

    def __init__(self,
                n_paloboost=10,
                distribution="gaussian",
                learning_rate=0.1,
                subsample=0.7,
                max_depth=3,
                n_estimators=100):
        self.n_paloboost = n_paloboost
        self.distribution = distribution
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.estimators = []

    def fit(self, X, y):
        n, m = X.shape
        idx = np.arange(n)
        self.estimators = []
        for i in range(self.n_paloboost):
            ridx = np.random.choice(idx, size=int(n*0.7), replace=False)
            X_i, y_i = X[ridx,:], y[ridx]
            est = PaloBoost(distribution=self.distribution,
                               learning_rate=self.learning_rate,
                                max_depth=self.max_depth,
                                n_estimators=self.n_estimators,
                                subsample=self.subsample)
            est.fit(X_i, y_i)
            self.estimators.append(est) 


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





