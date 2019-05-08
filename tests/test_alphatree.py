from __future__ import print_function

from bonsai.base.alphatree import AlphaTree
from bonsai.base.c45tree import C45Tree
from bonsai.base.ginitree import GiniTree

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import unittest

class AlphatreeTestCase(unittest.TestCase):

    def test_fit(self):

        n_samples = 100000
        max_depth = 4
        test_size = 0.2

        # Hastie_10_2
        # X_i ~ Gaussian
        # sum of X_i^2 > Chi-squire(10, 0.5) 9.34, then 1, otherwise -1
        X, y_org = make_hastie_10_2(n_samples=n_samples) 
        z = np.random.randn(n_samples)
        y = y_org * z
        y[y > 0] = 1
        y[y <= 0] = 0
        X = np.hstack((X, z.reshape(n_samples,1)))
        n, m = X.shape
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model = C45Tree(max_depth=max_depth)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        auc = roc_auc_score(y_test, y_hat)
        
        self.assertTrue(auc > 0.5)

        model = GiniTree(max_depth=max_depth)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        auc = roc_auc_score(y_test, y_hat)
        
        model = AlphaTree(alpha=3.0, max_depth=max_depth)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        self.assertTrue(auc > 0.5)


if __name__=="__main__":

    unittest.main()


