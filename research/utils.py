import time
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

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
                performance[name].append(
                            np.clip(r2_score(y_test, y_hat_i), 0, 1))
        else:
            for i in range(n_estimators):
                y_hat_i = model.predict(X_test, ntree_limit=i+1)
                performance[name].append(
                            np.clip(r2_score(y_test, y_hat_i), 0, 1))
    perf_df = pd.DataFrame(performance)
    perf_df['nEst'] = range(n_estimators)
    return perf_df

def simple_pp(X, 
        cat_max_k=20, 
        cat_ratio_thr=0.01, 
        num_min_std=1e-10,
        do_poly=True):

    # 0. divide columns to 1) categorical and 2) numeric features
    # 1. for categorical features, 
    #   1.1. replace the low count categories to more general categories
    #   1.2. dummy code categorical features
    #   1.3. remove the original features
    # 2. for numeric features,
    #   2.1. replace missing values with np.nan
    #   2.2. standardize features
    #   2.3. remove low variance features
    # 3. combine categorical and numeric features 
   
    # 0. 
    n, m = X.shape
    X_cat = X.select_dtypes(include=["object", "category"]).copy()
    X_num = X.select_dtypes(include=["number"]).copy()
    X_list = []

    print("- X.shape (n x m): {} by {}".format(X.shape[0], X.shape[1]))
    print("- m_[categorial features]: {}".format(X_cat.shape[1]))
    print("- m_[numeric features]: {}".format(X_num.shape[1]))

    # 1.1. 
    na_grp = "_na"
    misc_grp = "_others"
    for cname in X_cat.columns:
        x_cname = X_cat[cname].values
        x_cname[pd.isnull(x_cname)] = na_grp
        val_cnt = Counter(x_cname).most_common()
        val_under_thr = [val for val, cnt in val_cnt 
                        if (cnt+0.0)/n < cat_ratio_thr]
        if len(val_under_thr) > 0:
            x_cname[np.in1d(x_cname, val_under_thr)] = misc_grp
        if len(val_cnt) > cat_max_k:
            val_elig = [val for val, cnt in val_cnt[:(cat_max_k-1)]]
            x_cname[~np.in1d(x_cname, val_elig)] = misc_grp
        X_cat.loc[:,cname] = x_cname
    # 1.2.
    if len(X_cat.columns) > 0:
        X_list.append(pd.get_dummies(X_cat))
    
    col_num_exc = []
    for cname in X_num.columns:
        mean = X_num[cname].mean()
        std = X_num[cname].std()
        # 2.1.
        if std < num_min_std or np.isnan(std):
            col_num_exc.append(cname)
            continue
        # 2.2.
        X_num.loc[:,cname] = (X_num[cname].values - mean)/std
    # 2.3. 
    if len(col_num_exc) > 0:
        X_list.append(X_num.drop(col_num_exc, axis=1))
    else:
        X_list.append(X_num)

    X = pd.concat(X_list, axis=1)
   
    print("- m_[features after basic prep]: {}".format(X.shape[1]))

    if do_poly:
        X_imp = X.fillna(value=0)
        poly = PolynomialFeatures(degree=2, include_bias=False,
                                    interaction_only=True)
        X_poly = poly.fit_transform(X_imp)[:,X_imp.shape[1]:]
        X = pd.concat([X, pd.DataFrame(X_poly)], axis=1)

        print("- m_[features after poly prep]: {}".format(X.shape[1]))

    return X



