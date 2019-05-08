from bonsai.ensemble.paloforest import PaloForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from misc import make_hastie_11_2, make_freedman1_poly
from sklearn.metrics import brier_score_loss
import unittest


class PaloforestTestCase(unittest.TestCase):

    def test_cls(self):

        np.random.seed(1)
        n_samples = 10000
        test_size = 0.2
        n_est = 10
        max_depth = 5
        lr = 0.1

        X, y = make_hastie_11_2(n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size=test_size)

        model_org = PaloForest(distribution="bernoulli",
                                n_estimators=n_est, 
                                learning_rate=lr,
                                max_depth=max_depth,
                                calibrate=False)
        model_clb = PaloForest(distribution="bernoulli",
                                n_estimators=n_est, 
                                learning_rate=lr,
                                max_depth=max_depth,
                                calibrate=True)

        model_org.fit(X_train, y_train)
        y_hat = model_org.predict_proba(X_test)[:,1]
        auc_org = roc_auc_score(y_test, y_hat)
        brier_org = brier_score_loss(y_test, y_hat)

        model_clb.fit(X_train, y_train)
        y_hat = model_clb.predict_proba(X_test)[:,1]
        auc_clb = roc_auc_score(y_test, y_hat)
        brier_clb = brier_score_loss(y_test, y_hat)

        self.assertTrue(auc_org > 0.5)
        self.assertTrue(auc_clb > 0.5)
        self.assertTrue(brier_org > brier_clb)


if __name__=="__main__":

    unittest.main()

