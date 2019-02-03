import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

def get_cls_perf(models, X_train, y_train, X_test, y_test, n_estimators):
    performance = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        time_fit = time.time() - start
        print("  {0}: {1:.5f} sec...".format(name, time_fit))
        performance[name] = []        
        if "XGBoost" not in name:
            for i, y_hat_i in enumerate(model.staged_predict_proba(X_test)):
                performance[name].append(roc_auc_score(y_test, 
                                                        y_hat_i[:,1]))
        else:
            for i in range(n_estimators):
                y_hat_i = model.predict_proba(X_test, ntree_limit=i+1)
                performance[name].append(roc_auc_score(y_test, 
                                                        y_hat_i[:,1]))
    perf_df = pd.DataFrame(performance)
    perf_df['nEst'] = range(n_estimators)

    return perf_df

def get_reg_perf(models, X_train ,y_train, X_test, y_test, n_estimators):
    performance = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        time_fit = time.time() - start
        print("  {0}: {1:.5f} sec...".format(name, time_fit))
        performance[name] = []
        if "XGBoost" not in name:
            for i, y_hat_i in enumerate(model.staged_predict(X_test)):
                performance[name].append(np.clip(r2_score(y_test, y_hat_i), 0, 1))
        else:
            for i in range(n_estimators):
                y_hat_i = model.predict(X_test, ntree_limit=i+1)
                performance[name].append(np.clip(r2_score(y_test, y_hat_i), 0, 1))
    perf_df = pd.DataFrame(performance)
    perf_df['nEst'] = range(n_estimators)
    return perf_df


