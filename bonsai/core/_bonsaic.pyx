#cython: boundscheck=False
#cython: wrapround=False
#cython: cdivision=True

# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isnan

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def reorder(X, y, z, i_start, i_end, j_split, split_value, missing):
    return _reorder(X, y, z, i_start, i_end, j_split, split_value, missing)
 
cdef size_t _reorder(
        np.ndarray[DTYPE_t, ndim=2] X, 
        np.ndarray[DTYPE_t, ndim=1] y, 
        np.ndarray[DTYPE_t, ndim=1] z, 
        size_t i_start, 
        size_t i_end, 
        size_t j_split, 
        double split_value, 
        size_t missing):
    """
    - X: 2-d numpy array (n x m)
    - y: 1-d numpy array (n)
    - z: 1-d numpy array (n)
    - i_start: row index to start
    - i_end: row index to end
    - j_split: column index for the splitting variable
    - split_value: threshold
    """
    cdef size_t j
    cdef size_t m = X.shape[1]
    cdef size_t i_head = i_start
    cdef size_t i_tail = i_end - 1
    cdef size_t do_swap = 0

    with nogil:
        while i_head <= i_tail:

            if i_tail == 0: 
                # if tail is 'zero', should break
                # otherwise, segmentation fault, 
                # as size_t has no sign. 0 - 1 => huge number
                break
            
            do_swap = 0 
            if isnan(X[i_head,j_split]):
                if missing == 1: # send the missing to the right node
                    do_swap = 1
            else:
                if X[i_head,j_split] >= split_value:
                    do_swap = 1

            if do_swap == 1:
                # swap X rows
                for j in range(m):
                    X[i_head,j], X[i_tail,j] = X[i_tail,j], X[i_head,j]
                # swap y, z values
                y[i_head], y[i_tail] = y[i_tail], y[i_head]
                z[i_head], z[i_tail] = z[i_tail], z[i_head]
                # decrease the tail index
                i_tail -= 1
            else:
                # increase the head index
                i_head += 1

    return i_head


def sketch(np.ndarray[DTYPE_t, ndim=2] X not None, 
        np.ndarray[DTYPE_t, ndim=1] y not None, 
        np.ndarray[DTYPE_t, ndim=1] z not None, 
        np.ndarray[DTYPE_t, ndim=2] xdim not None, 
        np.ndarray[DTYPE_t, ndim=2] cnvs not None, 
        np.ndarray[DTYPE_t, ndim=2] cnvsn not None):

    # canvas --> (sketch) --> avc 
    # AVC: Attribute-Value Class group in RainForest
    _sketch(X, y, z, xdim, cnvs, cnvsn)
    return 0

cdef void _sketch(
        np.ndarray[DTYPE_t, ndim=2] X, 
        np.ndarray[DTYPE_t, ndim=1] y, 
        np.ndarray[DTYPE_t, ndim=1] z, 
        np.ndarray[DTYPE_t, ndim=2] xdim, 
        np.ndarray[DTYPE_t, ndim=2] cnvs, 
        np.ndarray[DTYPE_t, ndim=2] cnvsn):

    cdef size_t i, j, k, k_raw, k_tld
    cdef size_t n = X.shape[0]
    cdef size_t m = X.shape[1]
    cdef size_t n_cnvs = <size_t> cnvs.shape[0]/2
    cdef size_t n_bin
    cdef size_t xdim0 = <size_t> xdim[0, 4]
    cdef double k_prox
    cdef double y_i, z_i
    cdef double y_tot = 0.0
    cdef double z_tot = 0.0
    cdef double n_na, y_na, z_na

    # update E[y] & E[z]
    with nogil:

        for i in range(n):

            y_i = y[i]
            z_i = z[i]
            y_tot += y_i
            z_tot += z_i

            for j in range(m):

                #if xdim[j, 2] < 1e-12:
                #    continue
                if isnan(X[i, j]):
                    cnvsn[j, 1] += 1
                    cnvsn[j, 2] += y_i
                    cnvsn[j, 3] += z_i
                else:
                    k_prox = (X[i, j] - xdim[j, 1])/xdim[j, 2]
                    if k_prox < 0:
                        k_prox = 0
                    elif k_prox > xdim[j, 3] - 1:
                        k_prox = xdim[j, 3] - 1
                    k = <size_t> (k_prox + (xdim[j, 4] - xdim0)*2)
                    cnvs[k, 3] += 1
                    cnvs[k, 4] += y_i
                    cnvs[k, 5] += z_i

        # accumulate stats
        for j in range(m):
            n_bin = <size_t> xdim[j, 3]
            
            for k_raw in range(1, n_bin): 
                k = <size_t> (k_raw + (xdim[j, 4] - xdim0)*2)
                cnvs[k, 3] += cnvs[k-1, 3] 
                cnvs[k, 4] += cnvs[k-1, 4] 
                cnvs[k, 5] += cnvs[k-1, 5] 
                # fill the right node at the same time
                cnvs[k, 6] = n - cnvs[k, 3] - cnvsn[j, 1]
                cnvs[k, 7] = y_tot - cnvs[k, 4] - cnvsn[j, 2]
                cnvs[k, 8] = z_tot - cnvs[k, 5] - cnvsn[j, 3]

            # fill the right node
            k = <size_t> ((xdim[j, 4] - xdim0)*2)
            cnvs[k, 6] = n - cnvs[k, 3] - cnvsn[j, 1]
            cnvs[k, 7] = y_tot - cnvs[k, 4] - cnvsn[j, 2]
            cnvs[k, 8] = z_tot - cnvs[k, 5] - cnvsn[j, 3]

        # missing values
        for j in range(m):

            n_bin = <size_t> xdim[j, 3]
            n_na = cnvsn[j, 1]
            y_na = cnvsn[j, 2]
            z_na = cnvsn[j, 3]

            if n_na == 0:
                continue

            for k_raw in range(n_bin):
                k = <size_t> (k_raw + (xdim[j, 4] - xdim0)*2)
                k_tld = k + n_bin
                cnvs[k_tld, 3] = cnvs[k, 3]
                cnvs[k_tld, 4] = cnvs[k, 4]
                cnvs[k_tld, 5] = cnvs[k, 5]
                cnvs[k_tld, 6] = cnvs[k, 6]
                cnvs[k_tld, 7] = cnvs[k, 7]
                cnvs[k_tld, 8] = cnvs[k, 8]
                cnvs[k_tld, 9] = 1

                cnvs[k, 3] += n_na
                cnvs[k, 4] += y_na
                cnvs[k, 5] += z_na
                cnvs[k_tld, 6] += n_na
                cnvs[k_tld, 7] += y_na
                cnvs[k_tld, 8] += z_na

    # done _sketch

def apply_tree(tree_ind, tree_val, X, y, output_type):
    if output_type == "index":
        return _apply_tree0(tree_ind, tree_val, X, y)
    else:
        return _apply_tree1(tree_ind, tree_val, X, y)

# output index
cdef np.ndarray[DTYPE_t, ndim=1] _apply_tree0(
                            np.ndarray[np.int_t, ndim=2] tree_ind, 
                            np.ndarray[DTYPE_t, ndim=2] tree_val, 
                            np.ndarray[DTYPE_t, ndim=2] X, 
                            np.ndarray[DTYPE_t, ndim=1] y):
    # Initialize node/row indicies
    cdef size_t i, t
    cdef size_t n_samples = X.shape[0]

    with nogil:
        for i in range(n_samples):
            t = 0
            while tree_ind[t,0] < 0:
                if isnan(X[i, tree_ind[t,1]]):
                    if tree_ind[t,2]==0:
                        t = tree_ind[t,3] 
                    else:
                        t = tree_ind[t,4] 
                else:
                    if X[i,tree_ind[t,1]] < tree_val[t,0]:
                        t = tree_ind[t,3]
                    else:
                        t = tree_ind[t,4]
            y[i] = tree_ind[t,5]
    return y

# output y values
cdef np.ndarray[DTYPE_t, ndim=1] _apply_tree1(
                            np.ndarray[np.int_t, ndim=2] tree_ind, 
                            np.ndarray[DTYPE_t, ndim=2] tree_val, 
                            np.ndarray[DTYPE_t, ndim=2] X, 
                            np.ndarray[DTYPE_t, ndim=1] y):
    # Initialize node/row indicies
    cdef size_t i, t
    cdef size_t n_samples = X.shape[0]

    with nogil:
        for i in range(n_samples):
            t = 0
            while tree_ind[t,0] < 0:
                if isnan(X[i, tree_ind[t,1]]):
                    if tree_ind[t,2]==0:
                        t = tree_ind[t,3] 
                    else:
                        t = tree_ind[t,4] 
                else:
                    if X[i,tree_ind[t,1]] < tree_val[t,0]:
                        t = tree_ind[t,3]
                    else:
                        t = tree_ind[t,4]
            y[i] = tree_val[t,1]
    return y


