"""
This class implements the regression tree in CART that uses
the minimum variance criterion.
"""

# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

from bonsai.core.bonsaic import Bonsai
import numpy as np

class RegTree(Bonsai):
    def __init__(self, 
                max_depth=5, 
                min_samples_split=2,
                min_samples_leaf=1,
                min_varsum_decrease=0.0, 
                subsample=1.0,
                random_state=1234,
                **kwarg):

        self.max_depth=max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_varsum_decrease = min_varsum_decrease
 
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
            mu_l = avc[:,4]/n_l
            mu_r = avc[:,7]/n_r
            M2_l = avc[:,5]
            M2_r = avc[:,8]
            var_l = M2_l/n_l - mu_l*mu_l
            var_r = M2_r/n_r - mu_r*mu_r
            varsum = var_l * n_l + var_r * n_r
            best_idx = np.argsort(varsum)[0]
            best_varsum = varsum[best_idx]

            ss = {"selected": avc[best_idx,:],      # required for Bonsai
                "varsum@l": var_l[best_idx],        # required for RegTree
                "varsum@r": var_r[best_idx]}        # required for RegTree

            return ss

        def is_leaf(branch, branch_parent):

            varsum_dec = 1.0 + self.min_varsum_decrease
            if "varsum" in branch_parent:
                varsum_dec = branch_parent["varsum"] - branch["varsum"]
            if (branch["depth"] >= self.max_depth or 
                branch["n_samples"] < self.min_samples_split or
                varsum_dec < self.min_varsum_decrease):
                return True
            else:
                return False

        Bonsai.__init__(self, 
                        find_split, 
                        is_leaf,
                        subsample=subsample, 
                        random_state=random_state,
                        z_type="M2")


