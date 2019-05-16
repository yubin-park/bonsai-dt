import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from bonsai.ensemble.paloboost import PaloBoost
from bonsai.ensemble.gbm import GBM
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_hastie_10_2
from sklearn.preprocessing import PolynomialFeatures
import utils
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def get_friedman():
    n_samples = 10000
    noise = 5
    X, y = make_friedman1(n_samples=n_samples, noise=noise) 
    poly = PolynomialFeatures(degree=2, include_bias=False,
                            interaction_only=True)
    X = poly.fit_transform(X)
    print(X.shape)
    return X, y

def get_hastie():
    n_samples = 10000
    X, y_org = make_hastie_10_2(n_samples=n_samples) 
    z = np.random.randn(n_samples)
    y = y_org * z
    y[y > 0] = 1
    y[y <= 0] = 0
    r = np.random.rand(n_samples) < 0.2
    y[r] = 1 - y[r]
    X = np.hstack((X, z.reshape(n_samples, 1)))
    poly = PolynomialFeatures(degree=2, include_bias=False,
                            interaction_only=True)
    X = poly.fit_transform(X)
    print(np.mean(y))
    print(X.shape)
    return X, y

def get_losdata():
    data = pd.read_csv("data/featureSet3_48.csv")
    col_names = data.columns
    col_names_x = [cname for cname in col_names 
                    if cname not in ["RecordID", "Length_of_stay"]]
    X = utils.simple_pp(data[col_names_x], do_poly=True).values
    y = data["Length_of_stay"].values
    return X, y

def get_mortdata():
    data = pd.read_csv("data/featureSet3_48.csv")
    outcomes = pd.read_csv("data/Outcomes-a.txt")
    outcomes = outcomes[['RecordID', 'In-hospital_death']]
    data = pd.merge(data, outcomes, how='inner', on='RecordID')
    col_names = data.columns
    col_names_x = [cname for cname in col_names 
                    if cname not in ["RecordID", "Length_of_stay", 
                                        "In-hospital_death"]]
    X = utils.simple_pp(data[col_names_x], do_poly=True).values
    y = data["In-hospital_death"].values
    return X, y

def get_ca6hrdata():
    data0 = pd.read_csv("data/6Hr-train-1.csv")
    data1 = pd.read_csv("data/6Hr-test-1.csv")
    data = pd.concat([data0, data1], axis=0)
    y = data["ca"].values
    X = utils.simple_pp(data.drop(columns="ca")).values
    return X, y

def regtask(X, y, n_estimators, learning_rate, max_depth, n_btstrp, 
        has_missing=True):
    models = {"0. PaloBoost": PaloBoost(distribution="gaussian",
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth, 
                        subsample=0.7),
        "1. SGTB-Bonsai": GBM(distribution="gaussian",
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth, 
                        subsample=0.7),
        "2. XGBoost": XGBRegressor(
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate,
                    max_depth=max_depth, 
                    subsample=0.7)}
    if not has_missing:
        models["3. Scikit-Learn"] = GradientBoostingRegressor(
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        subsample=0.7)

    perf_df = pd.DataFrame(columns=["model", "value", "n_est", "b_idx"])
    for idx in range(n_btstrp):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size = 0.2,
                                                random_state=idx)
        df = utils.get_reg_perf(models, X_train, y_train, 
                                X_test, y_test, n_estimators)
        df['b_idx'] = idx
        perf_df = perf_df.append(df, sort=True)
    return perf_df

def clstask(X, y, n_estimators, learning_rate, max_depth, n_btstrp, 
            has_missing=True):
    models = {"0. PaloBoost": PaloBoost(distribution="bernoulli",
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth, 
                        subsample=0.7),
        "1. SGTB-Bonsai": GBM(distribution="bernoulli",
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth, 
                        subsample=0.7),
         "2. XGBoost": XGBClassifier(
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate,
                    max_depth=max_depth, 
                    subsample=0.7)}
    if not has_missing:
        models["3. Scikit-Learn"] = GradientBoostingClassifier(
                                        n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        max_depth=max_depth,
                                        subsample=0.7)
    perf_df = pd.DataFrame(columns=["model", "value", "n_est", "b_idx"])
    for idx in range(n_btstrp):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size = 0.2,
                                                random_state=idx)
        df = utils.get_cls_perf(models, X_train, y_train, 
                                X_test, y_test, n_estimators)
        df['b_idx'] = idx
        perf_df = perf_df.append(df, sort=True)
    return perf_df

def run(dataname, task, n_estimators, learning_rate, max_depth, 
            n_btstrp=10):

    np.random.seed(1)

    X, y = None, None
    has_missing = False
    if dataname == "friedman":
        X, y = get_friedman()
    elif dataname == "hastie":
        X, y = get_hastie()
    elif dataname == "los":
        X, y = get_losdata()
        has_missing = True
    elif dataname == "mort": 
        X, y = get_mortdata()
        has_missing = True
    elif dataname == "ca6hr": 
        X, y = get_ca6hrdata()

    perf_df = None
    if task == "reg": 
        perf_df = regtask(X, y, n_estimators, 
                            learning_rate, max_depth, 
                            n_btstrp, has_missing)
    else:
        perf_df = clstask(X, y, n_estimators, 
                            learning_rate, max_depth, 
                            n_btstrp, has_missing)

    perf_df.to_csv(("results/" + dataname +
                     "_{0}_{1}_{2}.csv".format(n_estimators,
                        learning_rate,max_depth)), index=False)

def run_aux(learning_rate, max_depth, n_estimators=200):

    X, y = get_friedman()
    model = PaloBoost(distribution="gaussian",
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth, 
                        subsample=0.7)
    model.fit(X, y)
    prune_df = pd.DataFrame(model.get_prune_stats())
    prune_df.columns = ["iteration", "nodes_pre", "nodes_post"]
    lr_df = pd.DataFrame(model.get_lr_stats())
    lr_df.columns = ["iteration", "lr"] 

    prune_df.to_csv("results/prune_{}_{}.csv".format(learning_rate, max_depth),
                    index=False)
    lr_df.to_csv("results/lr_{}_{}.csv".format(learning_rate, max_depth),
                    index=False)

if __name__ == "__main__":

    run_aux(0.1, 3)
    run_aux(0.5, 3)
    run_aux(1.0, 3)
    
    """
    run("friedman", "reg", 200, 0.1, 5)
    run("friedman", "reg", 200, 0.5, 5)
    run("friedman", "reg", 200, 1.0, 5)
    run("friedman", "reg", 200, 0.1, 4)
    run("friedman", "reg", 200, 0.5, 4)
    run("friedman", "reg", 200, 1.0, 4)
    run("friedman", "reg", 200, 0.1, 3)
    run("friedman", "reg", 200, 0.5, 3)
    run("friedman", "reg", 200, 1.0, 3)
    run("hastie", "cls", 200, 0.1, 5)
    run("hastie", "cls", 200, 0.5, 5)
    run("hastie", "cls", 200, 1.0, 5)
    run("hastie", "cls", 200, 0.1, 4)
    run("hastie", "cls", 200, 0.5, 4)
    run("hastie", "cls", 200, 1.0, 4)
    run("hastie", "cls", 200, 0.1, 3)
    run("hastie", "cls", 200, 0.5, 3)
    run("hastie", "cls", 200, 1.0, 3)
    run("los", "reg", 200, 0.1, 5)
    run("los", "reg", 200, 0.5, 5)
    run("los", "reg", 200, 1.0, 5)
    run("los", "reg", 200, 0.1, 4)
    run("los", "reg", 200, 0.5, 4)
    run("los", "reg", 200, 1.0, 4)
    run("los", "reg", 200, 0.1, 3)
    run("los", "reg", 200, 0.5, 3)
    run("los", "reg", 200, 1.0, 3)
    run("mort", "cls", 200, 0.1, 5)
    run("mort", "cls", 200, 0.5, 5)
    run("mort", "cls", 200, 1.0, 5)
    run("mort", "cls", 200, 0.1, 4)
    run("mort", "cls", 200, 0.5, 4)
    run("mort", "cls", 200, 1.0, 4)
    run("mort", "cls", 200, 0.1, 3)
    run("mort", "cls", 200, 0.5, 3)
    run("mort", "cls", 200, 1.0, 3)
    run("ca6hr", "cls", 200, 0.1, 5)
    run("ca6hr", "cls", 200, 0.5, 5)
    run("ca6hr", "cls", 200, 1.0, 5)
    run("ca6hr", "cls", 200, 0.1, 4)
    run("ca6hr", "cls", 200, 0.5, 4)
    run("ca6hr", "cls", 200, 1.0, 4)
    run("ca6hr", "cls", 200, 0.1, 3)
    run("ca6hr", "cls", 200, 0.5, 3)
    run("ca6hr", "cls", 200, 1.0, 3)
    """

