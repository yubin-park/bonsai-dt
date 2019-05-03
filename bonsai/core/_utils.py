"""
This module provide miscellaneous functions for the Bonsai class such as:
- get_child_branch
- setup_canvas
- reconstruct_tree
- ...
"""

# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

import numpy as np

PRECISION = 1e-12

def get_child_branch(ss, parent_branch, i_split, side):

    child_branch = {"eqs": list(parent_branch["eqs"])}
    ss_key = [key for key in ss.keys() if key[-2:] == side]
    ss_key += [key for key in ss.keys() if key[-2:] == "@m"]
    sidx = int(ss["selected"][0])
    svar = int(ss["selected"][1])
    sval = ss["selected"][2]
    missing = ss["selected"][9]
    parent_id = parent_branch["_id"]
    offset = 0

    if side == "@l":
        child_branch["i_start"] = parent_branch["i_start"]
        child_branch["i_end"] = i_split
        child_branch["_id"] = "{}::{}L".format(parent_id, sidx)
        child_branch["eqs"].append({"svar": svar, 
                                    "sval": sval, 
                                    #"sidx": sidx,
                                    "op": "<", 
                                    "missing": int(missing<0.5)})
    else:
        child_branch["i_start"] = i_split
        child_branch["i_end"] = parent_branch["i_end"]
        child_branch["_id"] = "{}::{}R".format(parent_id, sidx)
        child_branch["eqs"].append({"svar": svar, 
                                    "sval": sval, 
                                    #"sidx": sidx,
                                    "op": ">=", 
                                    "missing": int(missing>0.5)})
        offset = 3
   
    n_samples = ss["selected"][offset+3]
    sum_y = ss["selected"][offset+4]
    child_branch["n_samples"] = n_samples
    child_branch["y"] = sum_y / n_samples # mean(y)
    child_branch["depth"] = parent_branch["depth"] + 1

    for key in ss_key:
        key_new = key.replace(side,"")
        key_new = key_new.replace("@m","")
        child_branch[key_new] = ss[key]

    child_branch["y_lst"] = list(parent_branch["y_lst"]) 
    child_branch["y_lst"] += [child_branch["y"]]

    return child_branch

def get_xdim(X, n_hist_max):

    m = X.shape[1]

    # NOTE: x_min, x_max after removing outliers (top 1%, bottom 1%)
    x_min = np.nanpercentile(X, q=0.5, axis=0)
    x_max = np.nanpercentile(X, q=99.5, axis=0)
    #x_min = np.nanmin(np.ma.masked_invalid(X), axis=0)
    #x_max = np.nanmax(np.ma.masked_invalid(X), axis=0)

    xdim = np.zeros((m, 5), dtype=np.float, order="C")

    # Format of cavas row: [j, x_min, x_delta, n_bin, offset]
    for j in range(m):
        x_j = X[~np.isnan(X[:,j]),j]
        unique_values = np.unique(x_j)
        mask = np.logical_and(unique_values >= x_min[j], 
                            unique_values <= x_max[j])
        n_unique = np.sum(mask)
        if n_unique < n_hist_max: 
            n_bin = max(n_unique, 1)
        else:
            n_bin = n_hist_max
        xdim[j, 0] = j
        xdim[j, 1] = x_min[j]
        xdim[j, 2] = (x_max[j] - x_min[j])/n_bin # delta
        xdim[j, 3] = max(n_bin - 1, 1)
        if j > 0:
            xdim[j, 4] = xdim[j-1, 4] + xdim[j-1, 3] 

    return xdim       

def get_cnvsn(xdim): 
    # canvas for the missing values
    m, _ = xdim.shape
    cnvsn = np.zeros((m, 4))
    cnvsn[:,0] = np.arange(m)
    return cnvsn

def get_cnvs(xdim):
    m, _ = xdim.shape
    n = int(np.sum(xdim[:,3]))
    n_ = n * 2 # half for missing to the left, the other half to the right
    cnvs = np.zeros((n_, 10), dtype=np.float, order="C")
    cnvs[:,0] = np.arange(n_) 
    for j in range(m):
        x_min = xdim[j, 1]
        x_delta = xdim[j, 2]
        n_bin = int(xdim[j, 3])
        offset = int(xdim[j, 4]) * 2
        split_val = x_min
        for k in range(n_bin):
            i = offset + k
            split_val += x_delta 
            cnvs[i, 1] = j
            cnvs[i, 2] = split_val
            cnvs[(n_bin + i), 1] = j
            cnvs[(n_bin + i), 2] = split_val
            
    return cnvs

def reconstruct_tree(leaves):

    t_max = 0

    # binary search tree
    bst = {}     

    # output tree raw: [isleaf, svar, left, right, sval, out, index]
    # output tree row - 8 columns:
    #       integer section: [isleaf, svar, missing, left, right, index,
    #       float section:      sval, out]

    tree_raw = {} 
    for leaf in leaves:
        eqs = leaf["eqs"]
        node_ptr = bst 
        t = 0 # node index
        for depth, eq in enumerate(eqs):
            child_index = int(">="==eq["op"])
            svar = int(eq["svar"])
            #sidx = eq["sidx"]
            if child_index == 0:
                missing = int(eq["missing"]==0)
            else:
                missing = int(eq["missing"]==1)
            sval = float(eq["sval"])
            if "children" not in node_ptr:
                node_ptr["children"] = [{"t": t_max+1}, {"t": t_max+2}]
                tree_raw[t] = [-1, svar, missing, t_max+1, t_max+2, -1, 
                                sval, -1]
                t_max += 2
            node_ptr = node_ptr["children"][child_index]
            t = node_ptr["t"]
        tree_raw[t] = [1, -1, -1, -1, -1, leaf["index"], -1, leaf["y"]]

    tree_ind = np.zeros((t_max+1, 6), dtype=np.int, order="C")
    tree_val = np.zeros((t_max+1, 2), dtype=np.float, order="C")
    for t, node in tree_raw.items():
        tree_ind[t,:] = np.array(node[:6], dtype=np.int)
        tree_val[t,:] = np.array(node[6:], dtype=np.float)

    return tree_ind, tree_val

