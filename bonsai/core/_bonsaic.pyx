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
PRECISION = 1e-12

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
    
    while i_head <= i_tail:
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

cdef inline void erase_canvas(np.ndarray[DTYPE_t, ndim=2] canvas):
    canvas[:,3:] = 0

cdef inline void erase_canvas_na(np.ndarray[DTYPE_t, ndim=2] canvas_na):
    canvas_na[:,1:] = 0

def sketch(np.ndarray[DTYPE_t, ndim=2] X not None, 
        np.ndarray[DTYPE_t, ndim=1] y not None, 
        np.ndarray[DTYPE_t, ndim=1] z not None, 
        np.ndarray[DTYPE_t, ndim=2] canvas not None, 
        np.ndarray[DTYPE_t, ndim=2] canvas_dim not None, 
        np.ndarray[DTYPE_t, ndim=2] canvas_na not None, 
        size_t i_start, 
        size_t i_end):

    # canvas --sketch--> avc 
    # AVC: Attribute-Value Class group in RainForest
    _sketch(X, y, z, canvas, canvas_dim, canvas_na, i_start, i_end)

    valid = np.logical_and(canvas[:,3]>0, canvas[:,6]>0)
    avc = canvas[valid,:].copy()
 
    return avc

cdef void _sketch(
        np.ndarray[DTYPE_t, ndim=2] X, 
        np.ndarray[DTYPE_t, ndim=1] y, 
        np.ndarray[DTYPE_t, ndim=1] z, 
        np.ndarray[DTYPE_t, ndim=2] canvas, 
        np.ndarray[DTYPE_t, ndim=2] canvas_dim, 
        np.ndarray[DTYPE_t, ndim=2] canvas_na, 
        size_t i_start, 
        size_t i_end):

    cdef size_t i, j, k, k1
    cdef size_t n = i_end - i_start
    cdef size_t m = X.shape[1]
    cdef size_t n_canvas = <size_t> canvas.shape[0]/2
    cdef size_t offset, n_bin
    cdef double k_prox
    cdef double y_i, z_i
    cdef double y_tot = 0.0
    cdef double z_tot = 0.0
    cdef double n_na, y_na, z_na

    erase_canvas(canvas)
    erase_canvas_na(canvas_na)

    # update E[y] & E[z]
    with nogil:

        for i in range(i_start, i_end):
            y_i = y[i]
            z_i = z[i]
            y_tot += y_i
            z_tot += z_i
            for j in range(m):
                if canvas_dim[j, 2] < 1e-12:
                    continue
                elif isnan(X[i,j]):
                    canvas_na[j,1] += 1
                    canvas_na[j,2] += y_i
                    canvas_na[j,3] += z_i
                else:
                    k_prox = (X[i, j] - canvas_dim[j, 1])/canvas_dim[j, 2]
                    if k_prox < canvas_dim[j, 3]:
                        k = <size_t> (k_prox + canvas_dim[j, 4])
                        canvas[k, 3] += 1
                        canvas[k, 4] += y_i
                        canvas[k, 5] += z_i

        # accumulate stats
        for i in range(m):
            offset = <size_t>canvas_dim[i, 4]
            n_bin = <size_t>canvas_dim[i, 3]
            for j in range(1, n_bin): 
                k = offset + j
                canvas[k, 3] += canvas[k-1, 3] 
                canvas[k, 4] += canvas[k-1, 4] 
                canvas[k, 5] += canvas[k-1, 5] 

        # fill the right branch
        for i in range(n_canvas):
            j = <size_t> canvas[i,1]
            canvas[i, 6] = n - canvas[i, 3] - canvas_na[j, 1]
            canvas[i, 7] = y_tot - canvas[i, 4] - canvas_na[j, 2]
            canvas[i, 8] = z_tot - canvas[i, 5] - canvas_na[j, 3]

        # missing values

        for i in range(m):

            n_bin = <size_t> canvas_dim[i,3]
            offset = <size_t> canvas_dim[i,4] 
            n_na = canvas_na[i,1]
            y_na = canvas_na[i,2]
            z_na = canvas_na[i,3]

            if n_na > 0:
                for j in range(n_bin):
                    k = offset + j
                    k1 = k + n_canvas
                    canvas[k1,3] = canvas[k,3]
                    canvas[k1,4] = canvas[k,4]
                    canvas[k1,5] = canvas[k,5]
                    canvas[k1,6] = canvas[k,6]
                    canvas[k1,7] = canvas[k,7]
                    canvas[k1,8] = canvas[k,8]
                    canvas[k1,9] = 1

                    canvas[k,3] += n_na
                    canvas[k,4] += y_na
                    canvas[k,5] += z_na
                    canvas[k1,6] += n_na
                    canvas[k1,7] += y_na
                    canvas[k1,8] += z_na

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


