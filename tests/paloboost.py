from __future__ import print_function

from bonsai.ensemble.gbm import GBM
from bonsai.ensemble.paloboost import PaloBoost

from sklearn.datasets import make_friedman1
from sklearn.datasets import make_friedman2
from sklearn.datasets import make_friedman3
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import time


def test_regression():

    X, y = make_friedman1(n_samples=100000, noise=5) 
    n, m = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=0.5)

    models = {"palobst": PaloBoost(distribution="gaussian",
                            n_estimators=100,
                            learning_rate=1.0,
                            max_depth=4,
                            subsample=0.5),
            "gbm": GBM(distribution="gaussian",
                        n_estimators=100, 
                        learning_rate=1.0,
                        max_depth=4,
                        subsample=0.5),
            "sklearn": GradientBoostingRegressor(
                        n_estimators=100, 
                        learning_rate=1.0,
                        max_depth=4, 
                        subsample=0.5)}

    print("\n")
    print("# Test Regression")
    print("-----------------------------------------------------")
    print(" model_name     train_time     predict_time   rmse   ")
    print("-----------------------------------------------------")
    print(" {0:12}   {1:12}   {2:12}   {3:.5f}".format(
            "baseline", "-", "-", np.std(y_test)))

    for name, model in models.items():

        # Fit
        start = time.time()
        model.fit(X_train, y_train)
        time_fit = time.time() - start

        # Predict
        start = time.time()
        y_hat = model.predict(X_test)
        time_pred = time.time() - start

        # Error
        rmse = np.sqrt(np.mean((y_test - y_hat)**2))

        print(" {0:12}   {1:.5f} sec    {2:.5f} sec    {3:.5f}".format(
            name, time_fit, time_pred, rmse))

    print("-----------------------------------------------------")
    print("\n")

def test_classification():

    X, y = make_hastie_10_2(n_samples=1000) 
    y[y<0] = 0
    n, m = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=0.5)

    models = {"palobst": PaloBoost(distribution="bernoulli",
                            n_estimators=100, 
                            learning_rate=1.0,
                            max_depth=4,
                            subsample=0.5),
            "gbm": GBM(distribution="bernoulli",
                            n_estimators=100, 
                            learning_rate=1.0,
                            max_depth=4,
                            subsample=0.5),
            "sklearn": GradientBoostingClassifier(
                        n_estimators=100, 
                        learning_rate=1.0,
                        max_depth=4, 
                        subsample=0.5)}

    print("\n")
    print("# Test Classification")
    print("-----------------------------------------------------")
    print(" model_name     train_time     predict_time   auc    ")
    print("-----------------------------------------------------")
    print(" {0:12}   {1:12}   {2:12}   {3:.5f}".format(
            "baseline", "-", "-", 0.5))

    for name, model in models.items():

        # Fit
        start = time.time()
        model.fit(X_train, y_train)
        time_fit = time.time() - start

        # Predict
        start = time.time()
        y_hat = model.predict_proba(X_test)[:,1]
        time_pred = time.time() - start

        # Error
        auc = roc_auc_score(y_test, y_hat)

        print(" {0:12}   {1:.5f} sec    {2:.5f} sec    {3:.5f}".format(
            name, time_fit, time_pred, auc))

    print("-----------------------------------------------------")
    print("\n")


if __name__=="__main__":

    test_regression()
    #test_classification()

