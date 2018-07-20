from __future__ import print_function

from bonsai.base.regtree import RegTree
from bonsai.ensemble.randomforest import RandomForest
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
import numpy as np

def apply_tree(x, tree, use_varname=False):

    score = 0
    for leaf in tree:
        match = True
        for eq in leaf["eqs"]:
            svar = eq["svar"]
            sval = eq["sval"]
            if "<" == eq["op"]:
                if use_varname:
                    if x[eq["name"]] >= sval:
                        match = False 
                else:
                    if x[svar] >= sval:
                        match = False 
            else:
                if use_varname:
                    if x[eq["name"]] < sval:
                        match = False 
                else:
                    if x[svar] < sval:
                        match = False
            if not match:
                break
        if match:
            score = leaf["y"]
            break

    return score

def apply_randomforest(x, rf, use_varname=False):

    y = []
    for tree in rf:
        y.append(apply_tree(x, tree, use_varname))

    return np.mean(y)

def test():

    X, y = make_friedman1(n_samples=1000) 
    n, m = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                            test_size=0.2)

    reg_params = {"subsample": 0.8, 
                    "max_depth": 4}
    rf = RandomForest(base_estimator=RegTree,
                        base_params=reg_params,
                        n_estimators=20)

    print("\n")
    print("-----------------------------------------------------")

    # Fit
    rf.fit(X_train, y_train)

    # Predict
    y_hat_default = rf.predict(X_test)

    y_hat_script = np.zeros(y_test.shape[0])
    for i, x in enumerate(X_test):
        y_hat_script[i] = apply_randomforest(x, rf.dump())


    # Error
    match_rate = np.mean((y_hat_default - y_hat_script) < 1e-12)
    print("match_rate: {0:.5f} %".format(match_rate*100))
    print("-----------------------------------------------------")
    print("\n")

    

if __name__=="__main__":


    test()

