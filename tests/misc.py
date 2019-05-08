from sklearn.datasets import make_friedman1
from sklearn.datasets import make_hastie_10_2
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def make_hastie_11_2(n_samples):
    X, y_org = make_hastie_10_2(n_samples=n_samples) 
    z = np.random.randn(n_samples)
    y = y_org * z
    y[y > 0] = 1
    y[y <= 0] = 0
    r = np.random.rand(n_samples) < 0.2
    y[r] = 1 - y[r]
    X = np.hstack((X, z.reshape(n_samples, 1)))
    return X, y

def make_freedman1_poly(n_samples, noise=5):
    X, y = make_friedman1(n_samples=n_samples, noise=noise) 
    poly = PolynomialFeatures()
    X = poly.fit_transform(X)
    return X, y

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


