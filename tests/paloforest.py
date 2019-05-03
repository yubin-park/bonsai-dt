from bonsai.ensemble.gbm import GBM
from bonsai.ensemble.paloboost import PaloBoost
from bonsai.ensemble.paloforest import PaloForest

from sklearn.datasets import make_friedman1
from sklearn.datasets import make_friedman2
from sklearn.datasets import make_friedman3
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
import numpy as np
import time

def test_dumpload():

    X, y = make_hastie_10_2(n_samples=10000) 
    y[y<0] = 0
    n, m = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=0.5)

    model = PaloForest(distribution="bernoulli",
                            n_estimators=10, 
                            learning_rate=1.0,
                            max_depth=4,
                            subsample0=0.5,
                            calibrate=True)

    print("\n")
    print("# Test Dump/Load")
    print("-----------------------------------------------------")
    print(" model_name     train_time     predict_time   auc    ")
    print("-----------------------------------------------------")
    print(" {0:12}   {1:12}   {2:12}   {3:.5f}".format(
            "baseline", "-", "-", 0.5))

    # Fit
    start = time.time()
    model.fit(X_train, y_train)

    out = model.dump()
    model.load(out)

    time_fit = time.time() - start

    # Predict
    start = time.time()
    y_hat = model.predict_proba(X_test)[:,1]
    time_pred = time.time() - start

    auc = roc_auc_score(y_test, y_hat)

    print(" {0:12}   {1:.5f} sec    {2:.5f} sec    {3:.5f}".format(
        "palofrst", time_fit, time_pred, auc))

    print("-----------------------------------------------------")
    print("\n")



def test_classification():

    X, y = make_hastie_10_2(n_samples=10000) 
    y[y<0] = 0
    n, m = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=0.5)

    models = {"palofrst_org": PaloForest(distribution="bernoulli",
                            n_estimators=10, 
                            learning_rate=1.0,
                            max_depth=5,
                            subsample0=0.5,
                            calibrate=False),
            "palofrst_clb": PaloForest(distribution="bernoulli",
                            n_estimators=10, 
                            learning_rate=1.0,
                            max_depth=5,
                            subsample0=0.5,
                            calibrate=True)}

    print("\n")
    print("# Test Classification")
    print("-----------------------------------------------------")
    print(" model_name     train_time     auc       brier       ")
    print("-----------------------------------------------------")
    print(" {0:12}   {1:12}   {2:.5f}  {3:12}".format(
            "baseline", "-",0.5, "-"))

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
        brier = brier_score_loss(y_test, y_hat)

        print(" {0:12}   {1:.5f} sec    {2:.5f}  {3:.5f}".format(
            name, time_fit, auc, brier))

    print("-----------------------------------------------------")
    print("\n")


if __name__=="__main__":

    test_dumpload()
    test_classification()

