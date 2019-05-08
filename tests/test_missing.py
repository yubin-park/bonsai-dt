from bonsai.base.regtree import RegTree
from bonsai.base.xgbtree import XGBTree
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_friedman2
from sklearn.datasets import make_friedman3
from sklearn.model_selection import train_test_split
import numpy as np
import json
import time
import unittest


class MissingTestCase(unittest.TestCase):

    def test_missing(self):

        n_samples = 10000
        max_depth = 4
        test_size = 0.2
        missing_q = 20

        X, y = make_friedman1(n_samples=n_samples) 
        n, m = X.shape
        x_p = np.percentile(X, q=missing_q, axis=0)
        for j in range(X.shape[1]):
            X[X[:,j] < x_p[j],j] = np.nan
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model = RegTree(max_depth=max_depth)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        mse_baseline = np.sqrt(np.mean((y_test - np.mean(y_train))**2))
        mse_model = np.sqrt(np.mean((y_test - y_hat)**2))

        self.assertTrue(mse_model < mse_baseline)


if __name__=="__main__":

    unittest.main()


