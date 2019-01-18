"""
This module defines the Bonsai class and its basic templates.
The Bonsai class implments splitting data and constructing decision rules.
User need to provide two additional functions to complete the Bonsai class:
- find_split()
- is_leaf()
"""
# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

from bonsai.core._bonsaic import (
    reorder, 
    sketch,
    apply_tree)
from bonsai.core._utils import (
    reconstruct_tree,
    get_canvas_dim,
    setup_canvas_na,
    setup_canvas,
    get_child_branch)
import numpy as np
from scipy.special import expit
import time

class Bonsai:

    def __init__(self, 
                find_split,
                is_leaf, 
                n_hist_max=256, 
                subsample=1.0, 
                random_state=None,
                z_type="M2"):

        self.find_split = find_split # user-defined
        self.is_leaf = is_leaf       # user-defined
        self.n_hist_max = n_hist_max
        self.subsample = np.clip(subsample, 0.0, 1.0)
        self.random_state = random_state
        self.z_type = z_type

        self.leaves = []
        self.feature_importances_ = None
        self.n_features_ = 0
        self.tree_ind   = np.zeros((1,6), dtype=np.int)
        self.tree_val   = np.zeros((1,2), dtype=np.float)
        self.mask       = None
        self.canvas_dim = None
        self.canvas_na  = None
        self.canvas     = None

    def split_branch(self, X, y, z, branch):
        """Splits the data (X, y) into two children based on 
           the selected splitting variable and value pair.
        """

        i_start = branch["i_start"]
        i_end = branch["i_end"]

    
        # Get AVC-GROUP 
        avc = sketch(X, y, z, 
                    self.canvas, 
                    self.canvas_dim, 
                    self.canvas_na, 
                    i_start, i_end)

        if avc.shape[0] < 2:
            branch["is_leaf"] = True
            return [branch]

        # Find a split SS: selected split
        ss = self.find_split(avc)     
        if (not isinstance(ss, dict) or
            "selected" not in ss):
            branch["is_leaf"] = True
            return [branch]

        svar = ss["selected"][1]
        sval = ss["selected"][2]
        missing = ss["selected"][9]
        i_split = reorder(X, y, z, i_start, i_end, svar, sval, missing)

        if i_split==i_start or i_split==i_end:
            # NOTE: this condition may rarely happen due to 
            #       Python's floating point treatments.
            #       We just ignore this case, and stop the tree growth
            branch["is_leaf"] = True
            return [branch]

        left_branch = get_child_branch(ss, branch, i_split, "@l")
        left_branch["is_leaf"] = self.is_leaf(left_branch, branch)

        right_branch = get_child_branch(ss, branch, i_split, "@r")
        right_branch["is_leaf"] = self.is_leaf(right_branch, branch) 

        return [left_branch, right_branch]

    def grow_tree(self, X, y, z, branches):
        """Grows a tree by recursively partitioning the data (X, y)."""

        branches_new = []
        leaves_new = []
        for branch in branches:
            for child in self.split_branch(X, y, z, branch):
                if child["is_leaf"]:
                    leaves_new.append(child)
                else:
                    branches_new.append(child)
        return branches_new, leaves_new

    def fit(self, X, y, init_canvas=True): 
        """Fit a tree to the data (X, y)."""

        n, m = X.shape
        X = X.astype(np.float, order="C", copy=True)
        y = y.astype(np.float, order="C", copy=True)
        if self.z_type=="M2":
            z = np.square(y)
        elif self.z_type=="Hessian": # bernoulli hessian
            p = expit(y) 
            z = p * (1.0 - p)
        else:
            z = np.zeros(n) 

 
        if self.subsample < 1.0:
            np.random.seed(self.random_state)
            self.mask = (np.random.rand(n) < self.subsample)
            X = X[self.mask,:]
            y = y[self.mask]
            z = z[self.mask]
            n, m = X.shape
        else:
            self.mask = np.full(n, True, dtype=np.bool)

        self.n_features_ = m

        branches = [{"_id": "ROOT",
                    "is_leaf": False,
                    "depth": 0,
                    "eqs": [],
                    "i_start": 0,
                    "i_end": n,
                    "y": np.mean(y),
                    "y_lst": [], 
                    "n_samples": n}]

        if init_canvas:
            self.canvas_dim = get_canvas_dim(X, self.n_hist_max)
            self.canvas_na  = setup_canvas_na(self.canvas_dim.shape[0])
            self.canvas     = setup_canvas(self.canvas_dim)

        self.leaves = []
        if (self.canvas_dim is not None and 
            self.canvas is not None):
            while len(branches) > 0:
                branches, leaves_new = self.grow_tree(X, y, z, branches)
                self.leaves += leaves_new

        # integer index for leaves (from 0 to len(leaves))
        for i, leaf in enumerate(self.leaves): 
            leaf["index"] = i 
        self.get_feature_importances()
        self.tree_ind, self.tree_val = reconstruct_tree(self.leaves)

    def predict(self, X, output_type="response"):
        """Predict y by applying the trained tree to X."""
        X = X.astype(np.float)
        n, m = X.shape
        y = np.zeros(n, dtype=np.float)
        out = apply_tree(self.tree_ind, self.tree_val, X, y, output_type)
        return out 

    def init_canvas(self, X):
        self.canvas_dim = get_canvas_dim(X, self.n_hist_max)
        self.canvas = setup_canvas(self.canvas_dim)
        self.canvas_na = setup_canvas_na(self.canvas_dim.shape[0])

    def set_canvas(self, canvas_dim, canvas):
        self.canvas_dim = canvas_dim
        self.canvas = canvas
        self.canvas_na = setup_canvas_na(self.canvas_dim.shape[0])

    def get_canvas(self):
        return self.canvas_dim, self.canvas

    def is_stochastic(self):
        return self.subsample < 1.0

    def get_mask(self):
        return self.mask
    
    def get_oob_mask(self):
        """Returns a mask array for OOB samples"""
        return ~self.mask

    def get_ttab(self):
        """Returns tree tables (ttab). 
            ttab consists of tree_ind (np_array) and tree_val (np_array).
            tree_ind stores tree indices - integer array. 
            tree_val stores node values - float array.
        """
        return self.tree_ind, self.tree_val

    def dump(self, columns=[]):
        """Dumps the trained tree in the form of array of leaves"""
        n_col = len(columns)
        for leaf in self.leaves:
            for eq in leaf["eqs"]:
                if eq["svar"] < n_col:
                    eq["name"] = columns[int(eq["svar"])]
        return self.leaves

    def load(self, leaves, columns=None):
        """Loads a new tree in the form of array of leaves"""
        self.leaves = leaves
        self.get_feature_importances()
        self.tree_ind, self.tree_val = reconstruct_tree(self.leaves)
        return None

    def get_sibling_id(self, leaf_id):
        """Returns a sibling ID for the given leaf_id.
           Siblings are the nodes that are at the same level 
            with the same parent node.
        """
        sibling_id = None
        if leaf_id[-1]=="L":
            sibling_id = leaf_id[:-1] + "R"
        elif leaf_id[-1]=="R":
            sibling_id = leaf_id[:-1] + "L"
        sibling_leaf = [leaf for leaf in self.leaves 
                        if leaf["_id"]==sibling_id]
        if len(sibling_leaf) == 0:
            sibling_id = None
        return sibling_id 

    def get_sibling_pairs(self):
        """Returns an array of sibling pairs. 
            For more info, see the get_sibling_id
        """
        id2index = {leaf["_id"]:i for i, leaf in enumerate(self.leaves)}
        leaf_ids = [k for k in id2index.keys()]
        sibling_pairs = []
        while len(leaf_ids) > 0:
            leaf_id = leaf_ids.pop()
            sibling_id = self.get_sibling_id(leaf_id) 
            if sibling_id is not None:
                if sibling_id in leaf_ids:
                    leaf_ids.remove(sibling_id)
                sibling_pairs.append((id2index[leaf_id], 
                                      id2index[sibling_id]))
            else:
                sibling_pairs.append((id2index[leaf_id], None))
        return sibling_pairs
    
    def get_feature_importances(self):
        """Returns a modified feature importance.
            This formula takes into account of node coverage and leaf value.
            NOTE: This is different from regular feature importances that
                are used in RandomForests or GBM.
            For more info, please see the PaloBoost paper.
        """
        self.feature_importances_ = np.zeros(self.n_features_)
        cov = 0
        J = len(self.leaves)
        for j, leaf in enumerate(self.leaves):
            gamma_j = np.abs(leaf["y"])
            cov_j = leaf["n_samples"]
            cov += cov_j
            eff_j = cov_j*gamma_j
            for eq in leaf["eqs"]:
                self.feature_importances_[eq["svar"]] += eff_j
        self.feature_importances_ /= J
        self.feature_importances_ /= cov
        return self.feature_importances_ 

