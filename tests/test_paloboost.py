from bonsai.ensemble.paloboost import PaloBoost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from misc import make_hastie_11_2, make_freedman1_poly
import unittest

class PaloboostTestCase(unittest.TestCase):

    def test_cls(self):

        np.random.seed(1)
        n_samples = 10000
        test_size = 0.2
        n_est = 100
        max_depth = 7
        lr = 0.1

        X, y = make_hastie_11_2(n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model_palo = PaloBoost(distribution="bernoulli",
                                n_estimators=n_est, 
                                learning_rate=lr,
                                max_depth=max_depth)
        model_sklr = GradientBoostingClassifier(
                                n_estimators=n_est, 
                                learning_rate=lr,
                                max_depth=max_depth)


        model_palo.fit(X_train, y_train)
        y_hat = model_palo.predict_proba(X_test)[:,1]
        auc_palo = roc_auc_score(y_test, y_hat)

        model_sklr.fit(X_train, y_train)
        y_hat = model_sklr.predict_proba(X_test)[:,1]
        auc_sklr = roc_auc_score(y_test, y_hat)

        self.assertTrue(auc_palo > auc_sklr)

    def test_rgs(self):

        np.random.seed(1)
        n_samples = 10000
        test_size = 0.2
        n_est = 100
        max_depth = 7
        lr = 0.1

        X, y = make_freedman1_poly(n_samples=n_samples) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model_palo = PaloBoost(distribution="gaussian",
                                n_estimators=n_est,
                                learning_rate=lr,
                                max_depth=max_depth)
        model_sklr = GradientBoostingRegressor(
                            n_estimators=n_est, 
                            learning_rate=lr,
                            max_depth=max_depth)

        model_palo.fit(X_train, y_train)
        y_hat = model_palo.predict(X_test)
        rmse_palo = np.sqrt(np.mean((y_test - y_hat)**2))

        model_sklr.fit(X_train, y_train)
        y_hat = model_sklr.predict(X_test)
        rmse_sklr = np.sqrt(np.mean((y_test - y_hat)**2))

        self.assertTrue(rmse_palo < rmse_sklr)


if __name__=="__main__":

    unittest.main()

