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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import preprocess as pp
import utils
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def get_losdata():
    data = pd.read_csv("data/featureSet3_48.csv")
    col_names = data.columns
    col_names_x = [cname for cname in col_names 
                    if cname not in ["RecordID", "Length_of_stay"]]
    X = pp.simple_pp(data[col_names_x]).values
    y = data["Length_of_stay"].values
    print("- Number of features after poly: {}".format(X.shape[1]))
    return X, y

def get_mortdata():
    data = pd.read_csv("data/featureSet3_48.csv")
    outcomes = pd.read_csv("data/outcomes-a.txt")
    outcomes = outcomes[['RecordID', 'In-hospital_death']]
    data = pd.merge(data, outcomes, how='inner', on='RecordID')
    col_names = data.columns
    col_names_x = [cname for cname in col_names 
                    if cname not in ["RecordID", "Length_of_stay", 
                                        "In-hospital_death"]]
    X = pp.simple_pp(data[col_names_x]).values
    y = data["In-hospital_death"].values
    print("- Number of features after poly: {}".format(X.shape[1]))
    return X, y

def get_ca6hrdata():
    data0 = pd.read_csv("data/6Hr-train-1.csv")
    data1 = pd.read_csv("data/6Hr-test-1.csv")
    data = pd.concat([data0, data1], axis=0)
    y = data["ca"].values
    X = pp.simple_pp(data.drop(columns="ca")).values
    return X, y

def regtask(X, y, n_estimators, learning_rate, max_depth, n=30):
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
    perf_df = pd.DataFrame(columns=["0. PaloBoost", "1. SGTB-Bonsai",
                                    "2. XGBoost", "nEst", "idx"])
    for idx in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size = 0.5,
                                                random_state=idx)
        df = utils.get_reg_perf(models, X_train, y_train, 
                                        X_test, y_test, n_estimators)
        df['idx'] = idx
        perf_df = perf_df.append(df)
    return perf_df

def clstask(X, y, n_estimators, learning_rate, max_depth, n=30):
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
    perf_df = pd.DataFrame(columns=["0. PaloBoost", "1. SGTB-Bonsai",
                                    "2. XGBoost", "nEst", "idx"])
    for idx in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size = 0.5,
                                                random_state=idx)
        df = utils.get_cls_perf(models, X_train, y_train, 
                                        X_test, y_test, n_estimators)
        df['idx'] = idx
        perf_df = perf_df.append(df)
    return perf_df

def run(dataname, task, n_estimators, learning_rate, max_depth):

    np.random.seed(0)
    subsample = 0.7

    X, y = None, None
    if dataname == "los":
        X, y = get_losdata()
    elif dataname == "mort": 
        X, y = get_mortdata()
    elif dataname == "ca6hr": 
        X, y = get_ca6hrdata()

    perf_df = None
    if task == "reg": 
        perf_df = regtask(X, y, n_estimators, learning_rate, max_depth)
    else:
        perf_df = clstask(X, y, n_estimators, learning_rate, max_depth)

    perf_df.to_csv(("results/" + dataname +
                     "_{0}_{1}_{2}_{3}.csv".format(n_estimators,
                        learning_rate,max_depth,subsample)), index=False)

if __name__ == "__main__":

    run("los", "reg", 500, 0.1, 5)
    run("los", "reg", 500, 0.5, 5)
    run("los", "reg", 500, 1.0, 5)
    run("los", "reg", 500, 0.1, 7)
    run("los", "reg", 500, 0.5, 7)
    run("los", "reg", 500, 1.0, 7)

    run("mort", "cls", 500, 0.1, 5)
    run("mort", "cls", 500, 0.5, 5)
    run("mort", "cls", 500, 1.0, 5)
    run("mort", "cls", 500, 0.1, 7)
    run("mort", "cls", 500, 0.5, 7)
    run("mort", "cls", 500, 1.0, 7)

    run("ca6hr", "cls", 500, 0.1, 5)
    run("ca6hr", "cls", 500, 0.5, 5)
    run("ca6hr", "cls", 500, 1.0, 5)
    run("ca6hr", "cls", 500, 0.1, 7)
    run("ca6hr", "cls", 500, 0.5, 7)
    run("ca6hr", "cls", 500, 1.0, 7)
 



