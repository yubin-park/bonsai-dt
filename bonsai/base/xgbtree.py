"""
This class implements the XGBoost base tree that appeared in:
- https://arxiv.org/abs/1603.02754 
"""

# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

from bonsai.core.bonsaic import Bonsai
import numpy as np

class XGBTree(Bonsai):
    def __init__(self, 
                max_depth=5, 
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=1.0,
                subsample_splts=1.0,
                reg_lambda=1e-2,            # regularization
                obj_tolerance=1e-2,
                random_state=1234,
                distribution="gaussian",
                n_jobs=-1,
                **kwarg):
        
        self.max_depth=max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.distribution = distribution
        self.subsample_splts = np.clip(subsample_splts, 0.0, 1.0)
        self.obj_tolerance = obj_tolerance

        def find_split(avc):

            if avc.shape[0] == 0:
                return None

            valid_splits = np.logical_and(
                            avc[:,3] > self.min_samples_leaf,
                            avc[:,6] > self.min_samples_leaf)
            avc = avc[valid_splits,:]

            if self.subsample_splts < 1.0:
                n_avc = avc.shape[0]
                mask = np.random.rand(n_avc) < self.subsample_splts
                avc = avc[mask,:]

            if avc.shape[0] == 0:
                return None

            if self.distribution=="bernoulli":
                h_l = avc[:,5] 
                h_r = avc[:,8]
            else: 
                h_l = avc[:,3]
                h_r = avc[:,6]
            g_l = avc[:,4] 
            g_r = avc[:,7] 
            obj = g_l*g_l/(h_l+self.reg_lambda) 
            obj = obj + g_r*g_r/(h_r+self.reg_lambda)


            y_l = g_l/(h_l+self.reg_lambda)
            y_r = g_r/(h_r+self.reg_lambda)

            best_idx = np.argsort(obj)[-1]

            if (self.distribution == "bernoulli" and 
                obj[best_idx] < self.obj_tolerance):
                return None

            ss = {"selected": avc[best_idx,:],
                  "y@l": y_l[best_idx],
                  "y@r": y_r[best_idx]}

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
                        n_jobs = n_jobs,
                        z_type="Hessian")

    def prune(self, X, y, y_hat, nu, max_iter=1):

        # prune status
        # - 0: don't know if it passed the test
        # - 1: passed the test

        def loss(y, y_hat):
            if self.distribution == "gaussian":
                return np.mean((y-y_hat)**2)
            elif self.distribution == "bernoulli":
                return np.mean(-2.0*(y*y_hat - np.logaddexp(0.0, y_hat)))
            else:
                return np.mean((y-y_hat)**2)

        def merge_nodes(leaf_j, leaf_k):
            leaf = leaf_j.copy()
            leaf["i_start"] = np.min([leaf_j["i_start"], 
                                        leaf_k["i_start"]])   
            leaf["i_end"] = np.max([leaf_j["i_end"], leaf_k["i_end"]]) 
            leaf["_id"] = "::".join(leaf["_id"].split("::")[:-1])
            leaf["eqs"] = leaf["eqs"][:-1]
            leaf["depth"] = leaf["depth"] - 1
            leaf["n_samples"] = leaf_j["n_samples"] + leaf_k["n_samples"]
            leaf["y_lst"] = list(leaf_j["y_lst"][:-1])
            leaf["y"] = leaf["y_lst"][-1]
            leaf["prune_status"] = 0
            return leaf

        def prune0(X, y, y_hat, nu):

            oob_mask = self.get_oob_mask()
            t = self.predict(X, "index")
            leaves = self.dump()
            sibling_pairs = self.get_sibling_pairs()

            leaves_new = []
            for j, k in sibling_pairs:

                if "prune_status" not in leaves[j]:
                    leaves[j]["prune_status"] = 0
                if k is not None and "prune_status" not in leaves[k]:
                    leaves[k]["prune_status"] = 0

                if leaves[j]["depth"] < 2:
                    leaves[j]["prune_status"] = 1
                    if k is not None and leaves[k]["depth"] < 2:
                        leaves[k]["prune_status"] = 1
                    continue

                if k is None:
                    leaves_new.append(leaves[j])
                    continue
               
                if (leaves[j]["prune_status"]==1 and
                    leaves[k]["prune_status"]==1):
                    continue

                mask_j = np.logical_and((t==j), oob_mask)
                mask_k = np.logical_and((t==k), oob_mask)
                gamma_j = leaves[j]["y"]
                gamma_k = leaves[k]["y"]

                do_merge = False
                if (np.sum(mask_j) == 0 or np.sum(mask_k) == 0):
                    do_merge = True
                else:
                    y_j = y[mask_j]
                    y_hat_j = y_hat[mask_j]
                    y_k = y[mask_k]
                    y_hat_k = y_hat[mask_k]
                    y_hat_j_new = y_hat_j + gamma_j * nu
                    y_hat_k_new = y_hat_k + gamma_j * nu

                    if loss(y_j, y_hat_j) < loss(y_j, y_hat_j_new):
                        do_merge = True
                    elif loss(y_k, y_hat_k) < loss(y_k, y_hat_k_new):
                        do_merge = True

                if do_merge:
                    leaf_m = merge_nodes(leaves[j], leaves[k])
                    mask_m = np.logical_or(mask_j, mask_k)
                    if np.sum(mask_m) > 0:
                        gamma_m = leaf_m["y"] 
                        y_m = y[mask_m]
                        y_hat_m = y_hat[mask_m]
                        y_hat_m_new = y_hat_m + gamma_m*nu
                        if loss(y_m, y_hat_m) > loss(y_m, y_hat_m_new):
                            leaf_m["prune_status"] = 1
                    leaves_new.append(leaf_m)
                else:
                    leaves[j]["prune_status"] = 1
                    leaves[k]["prune_status"] = 1
                    leaves_new.append(leaves[j])
                    leaves_new.append(leaves[k])

            for i, leaf in enumerate(leaves_new):
                leaf["index"] = i 

            self.load(leaves_new) 

            pruned = np.sum([leaf["prune_status"] for leaf in leaves_new])

            return pruned == len(leaves_new)

        if max_iter < 0:
            max_iter = np.log2(len(self.dump())) - 2

        do_pruning = True
        i = 0
        while do_pruning:
            nomore_pruning = prune0(X, y, y_hat, nu)
            i += 1
            if nomore_pruning or i >= max_iter:
                do_pruning = False

        # DONE PRUNING

