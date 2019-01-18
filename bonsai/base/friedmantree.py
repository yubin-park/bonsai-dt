"""
This class implements a tree with the Friedman's splitting criterion
that appeared in: 
- https://statweb.stanford.edu/~jhf/ftp/trebst.pdf
FYI, this is the default splitting criterion for Scikit-Learn GBM.
"""

# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

from bonsai.core.bonsaic import Bonsai
import numpy as np

class FriedmanTree(Bonsai):
    def __init__(self, 
                max_depth=5, 
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=1.0,
                random_state=1234,
                **kwarg):

        self.max_depth=max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
 
        def find_split(avc):

            if avc.shape[0] == 0:
                return None

            valid_splits = np.logical_and(
                            avc[:,3] > self.min_samples_leaf,
                            avc[:,6] > self.min_samples_leaf)
            avc = avc[valid_splits,:]

            if avc.shape[0] == 0:
                return None

            n_l = avc[:,3]
            n_r = avc[:,6]
            y_hat_l = avc[:,4]/n_l
            y_hat_r = avc[:,7]/n_r

            diff = y_hat_l - y_hat_r
            diff2 = diff*diff
            friedman_score = n_l * n_r / (n_l + n_r) * diff2
            best_idx = np.argsort(friedman_score)[-1]
            ss = {"selected": avc[best_idx,:]}
            return ss

        def is_leaf(branch, branch_parent):

            if (branch["depth"] >= self.max_depth or 
                branch["n_samples"] < self.min_samples_split):
                return True
            else:
                return False

        Bonsai.__init__(self, 
                        find_split, 
                        is_leaf,
                        subsample=subsample, 
                        random_state=random_state,
                        z_type="M2")


