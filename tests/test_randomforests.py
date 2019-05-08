from bonsai.base.regtree import RegTree
from bonsai.ensemble.randomforests import RandomForests
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
import numpy as np
import time
import unittest

class RandomForestsTestCase(unittest.TestCase):

    def test_fit(self):

        n_samples = 10000
        test_size = 0.2
        n_est = 10
        reg_params = {"subsample": 0.7, 
                        "max_depth": 4}

        X, y = make_friedman1(n_samples=n_samples) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model = RandomForests(base_estimator=RegTree,
                                base_params=reg_params,
                                n_estimators=n_est)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        mse_rf = np.mean((y_test - y_hat)**2)
        mse_baseline = np.mean((y_test - np.mean(y_train))**2)

        self.assertTrue(mse_rf < mse_baseline)


if __name__=="__main__":

    unittest.main()


