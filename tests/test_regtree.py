from bonsai.base.regtree import RegTree
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_friedman2
from sklearn.datasets import make_friedman3
from sklearn.model_selection import train_test_split
import numpy as np
import unittest

class RegtreeTestCase(unittest.TestCase):

    def test_fit(self):

        n_samples = 10000
        max_depth = 4
        test_size = 0.2

        X, y = make_friedman1(n_samples=n_samples) 
        n, m = X.shape
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model = RegTree(max_depth=max_depth)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        mse_baseline = np.mean((y_test - np.mean(y_train))**2)
        mse_regtree = np.mean((y_test - y_hat)**2)

        self.assertTrue(mse_baseline > mse_regtree)


if __name__=="__main__":

    unittest.main()


