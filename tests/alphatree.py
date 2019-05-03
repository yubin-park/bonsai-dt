from __future__ import print_function

from bonsai.base.alphatree import AlphaTree
from bonsai.base.c45tree import C45Tree
from bonsai.base.ginitree import GiniTree

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import time

def test():

    X, y = make_hastie_10_2(n_samples=1000000) 
    y[y==-1.0] = 0.0        # AlphaTree accepts [0, 1] not [-1, 1]
    poly = PolynomialFeatures(degree=2)
    X = poly.fit_transform(X)
    n, m = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=0.2)
    print(X.shape) 
    depth = 7
    models = {"alpha_1-c45": C45Tree(max_depth=depth),
            "alpha_2-cart": GiniTree(max_depth=depth),
            "alpha_3": AlphaTree(alpha=3.0, max_depth=depth),
            "sklearn": DecisionTreeClassifier(max_depth=depth)}

    print("\n")
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
        y_hat = model.predict(X_test)
        time_pred = time.time() - start

        # Error
        auc = roc_auc_score(y_test, y_hat)
        print(" {0:12}   {1:.5f} sec    {2:.5f} sec    {3:.5f}".format(
            name, time_fit, time_pred, auc))

    print("-----------------------------------------------------")
    print("\n")


if __name__=="__main__":

    test()


