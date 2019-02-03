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
import preprocessing as pp
import eval_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile",
                        help="filename for performance (csv)")
    parser.add_argument("-n", type=int, default=200,
                        help="number of estimators")
    parser.add_argument("-lr", type=float, default=1.0,
                        help="learning rate")
    parser.add_argument("-sub", type=float, default=0.7,
                        help="subsample rate")
    parser.add_argument("-depth", type=int, default=5,
                        help="subsample rate")
    args = parser.parse_args()

    # Parameters
    n_estimators = args.n
    learning_rate = args.lr # 1.0, 0.5, 0.1
    test_size = 0.7  # 30% training, 70% test - to highlight the overfitting aspect of the models
    subsample = args.sub
    max_depth = args.depth

    data = pd.read_csv("data/6Hr-data.csv")
    y = data["ca"].values
    X = pp.simple_pp(data.drop(columns="ca")).values

    print("- Avg(y): {}, Std(y): {}".format(np.mean(y), np.std(y)))

    models = {"0. PaloBoost    ": PaloBoost(distribution="bernoulli",
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth, 
                        subsample=subsample),
        "1. SGTB-Bonsai": GBM(distribution="bernoulli",
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth, 
                        subsample=subsample),
         "2. XGBoost      ": XGBClassifier(
                    n_estimators=n_estimators, 
                    learning_rate=learning_rate,
                    max_depth=max_depth, 
                    subsample=subsample),
         "3. Scikit-Learn ": GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample)}
    boostPerf = pd.DataFrame(columns=["0. PaloBoost    ", "1. SGTB-Bonsai",
                                    "2. XGBoost      ", "3. Scikit-Learn ",
                                    "nEst", "idx"])
    for idx in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size = test_size,
                                                random_state=idx)
        perf_df = eval_utils.get_cls_perf(models, X_train, y_train, 
                                        X_test, y_test, n_estimators)
        perf_df['idx'] = idx
        boostPerf = boostPerf.append(perf_df)
    # store it to the file
    boostPerf.to_csv((args.outfile +
                        "_{0}_{1}_{2}_{3}.csv".format(n_estimators,
                        learning_rate,max_depth,subsample)), index=False)
    # spit out the highest max for each class
    tmpDF = boostPerf.groupby(['idx']).max()
    print(tmpDF.mean())

if __name__ == "__main__":
    main()
