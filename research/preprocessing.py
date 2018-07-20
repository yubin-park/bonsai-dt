from __future__ import print_function
import numpy as np
import pandas as pd
from collections import Counter

def simple_pp(X):

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
   
    # Parameters
    cat_max_k = 20
    cat_ratio_thr = 0.01
    num_min_std = 1e-10

    # 0. 
    n, m = X.shape
    X_cat = X.select_dtypes(include=["object", "category"]).copy()
    X_num = X.select_dtypes(include=["number"]).copy()
    X_list = []

    print("- X.shape: {} by {}".format(X.shape[0], X.shape[1]))
    print("- Number of categorial features: {}".format(X_cat.shape[1]))
    print("- Number of numeric features: {}".format(X_num.shape[1]))

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
   
    print("- Number of features after pp: {}".format(X.shape[1]))
 
    return X

def test():

    data = pd.read_csv("../data/kaggle-bnp/train.csv")
    col_names = data.columns
    col_names_x = [cname for cname in col_names 
                    if cname not in ["ID", "target"]]
    X = simple_pp(data[col_names_x])
    y = data["target"].values

    return 0
    
if __name__=="__main__":

    test()
