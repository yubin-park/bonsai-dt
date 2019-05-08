from bonsai.ensemble.gbm import GBM
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_friedman2
from sklearn.datasets import make_friedman3
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import unittest

class GBMTestCase(unittest.TestCase):

    def test_fit(self):

        n_samples = 10000
        test_size = 0.2
        max_depth = 3
        lr = 0.1
        n_est = 100

        X, y = make_friedman1(n_samples=n_samples) 
        n, m = X.shape
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model = GBM(distribution="gaussian",
                                n_estimators=n_est, 
                                learning_rate=lr,
                                max_depth=max_depth)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        mse_gbm = np.mean((y_test - y_hat)**2)
        mse_baseline = np.mean((y_test - np.mean(y_train))**2)

        self.assertTrue(mse_gbm < mse_baseline)


if __name__=="__main__":

    unittest.main()

