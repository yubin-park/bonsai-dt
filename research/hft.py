from bonsai.base.lhf import LHF
from bonsai.base.alphatree import AlphaTree
from bonsai.base.regtree import RegTree
from bonsai.base.xgb import XGBTree
from bonsai.ensemble.randomforest import RandomForest
from bonsai.ensemble.gbm import GBM
from bonsai.ensemble.resgbm import ResGBM
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

def run_hft(currency):

    data_fn = "../data/gdax/{}-USD_Xy.csv".format(currency)
    model_fn = "../dumps/hft_{}.json".format(currency)

    data = pd.read_csv(data_fn)
    data = data[data["close_smavg12"] > 1.0]
    data = data[data["z"] < 1.0]
    columns = data.columns.tolist()
    y = data["y"].values
    X = data.drop(["y", "z"], axis=1).values
    n, m = X.shape
    tt_split = int(n*0.9)
    X_train = X[:tt_split,:]
    X_test = X[tt_split:,:]
    y_train = y[:tt_split]
    y_test = y[tt_split:] 

    reg_params = {"max_depth": 4, 
                    "reg_lambda": 1e1, 
                    "subsample": 0.5}
    gbm = ResGBM(base_estimator=XGBTree,
                base_params=reg_params,
                distribution="Bernoulli",
                n_estimators=20, 
                learning_rate=1)
    gbm.fit(X_train, y_train)
    y_hat = gbm.predict(X_test)
 
    prec, recall, thr = precision_recall_curve(y_test, y_hat)
    fig = plt.figure()
    plt.plot(recall, prec, "b")
    plt.savefig("../dumps/hft_{}.png".format(currency))
    plt.close(fig)
    cov6 = np.mean(y_hat > 0.6)
    cov7 = np.mean(y_hat > 0.7)
    cov8 = np.mean(y_hat > 0.8)
    print(" {0:12}   {1:.5f}        {2:.5f}        {3:.5f}".format(
        currency, cov6, cov7, cov8))
    json.dump(gbm.dump(columns), 
                fp=open(model_fn, "w"),
                indent=2, 
                sort_keys=True)
    return 0

def run():

    print("\n")
    print("-------------------------------------------------------")
    print(" currency       cov (0.6)      cov (0.7)     cov (0.8) ")
    print("-------------------------------------------------------")

    currencies = ["BTC", "BCH", "ETH", "LTC"]

    for currency in currencies:
        run_hft(currency)

    print("-------------------------------------------------------")
    print("\n")

if __name__=="__main__":

    run()



