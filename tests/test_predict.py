from bonsai.base.regtree import RegTree
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
import numpy as np
import unittest
from misc import apply_tree

class PredictTestCase(unittest.TestCase):

    def test_predict(self):

        n_samples = 1000
        max_depth = 4
        test_size = 0.2

        X, y = make_friedman1(n_samples=n_samples) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model = RegTree(max_depth=max_depth)

        model.fit(X_train, y_train)
        y_hat_default = model.predict(X_test)
        y_hat_script = np.zeros(y_test.shape[0])
        for i, x in enumerate(X_test):
            y_hat_script[i] = apply_tree(x, model.dump())

        match_rate = np.mean((y_hat_default - y_hat_script) < 1e-12)
        self.assertAlmostEqual(match_rate, 1.0)


if __name__=="__main__":

    unittest.main()

