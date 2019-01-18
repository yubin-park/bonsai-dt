"""
This class implements a modified version of Alpha Tree that appeared in
- the origial: https://ieeexplore.ieee.org/document/6399474/
- closer to the implementation: https://arxiv.org/abs/1606.05325
"""

# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

from bonsai.core.bonsaic import Bonsai
import numpy as np

PRECISION = 1e-12

class AlphaTree(Bonsai):

    def __init__(self,
                alpha=1.0, 
                max_depth=5, 
                min_samples_split=2,
                min_samples_leaf=1, 
                **kwarg):

        self.alpha = alpha
        self.max_depth=max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        def find_split(avc):

            if avc.shape[0] == 0:
                return None

            valid_splits = np.logical_and(
                            avc[:,3] > max(self.min_samples_leaf, 1),
                            avc[:,6] > max(self.min_samples_leaf, 1))
            avc = avc[valid_splits,:]
            if avc.shape[0] == 0:
                return None
    
            n_l = avc[:,3]
            n_r = avc[:,6]
            p_y_l = (avc[:,4] + 1.0)/(n_l + 2.0)
            p_y_r = (avc[:,7] + 1.0)/(n_r + 2.0)

            p_x_l = n_l / (n_l + n_r)
            p_x_r = n_r / (n_l + n_r)

            p_y_l[p_y_l < PRECISION] = PRECISION
            p_y_r[p_y_r < PRECISION] = PRECISION
            p_y_l[p_y_l > 1.0-PRECISION] = 1.0-PRECISION
            p_y_r[p_y_r > 1.0-PRECISION] = 1.0-PRECISION

            gain = np.zeros(avc.shape[0])

            if self.alpha == 1.0: # Information Gain
                gain += (p_x_l * p_y_l * np.log(p_y_l))
                gain += (p_x_l * (1.0-p_y_l) * np.log(1.0-p_y_l))
                gain += (p_x_r * p_y_r * np.log(p_y_r))
                gain += (p_x_r * (1.0-p_y_r) * np.log(1.0-p_y_r))
            elif self.alpha == 0.0:
                gain -= (p_x_l * np.log(p_y_l))
                gain -= (p_x_l * np.log(1.0 - p_y_l))
                gain -= (p_x_r * np.log(p_y_r))
                gain -= (p_x_r * np.log(1.0 - p_y_r))
            else:
                gain -= (p_x_l * np.power(p_y_l, self.alpha))
                gain -= (p_x_l * np.power(1.0 - p_y_l, self.alpha))
                gain -= (p_x_r * np.power(p_y_r, self.alpha))
                gain -= (p_x_r * np.power(1.0 - p_y_r, self.alpha))
                gain = gain / self.alpha / (1.0 - self.alpha)

            best_idx = np.argsort(gain)[-1]

            return {"selected": avc[best_idx,:]}

        def is_leaf(branch, branch_parent):

            if (branch["depth"] >= self.max_depth or 
                branch["n_samples"] < self.min_samples_split):
                return True
            else:
                return False

        Bonsai.__init__(self, find_split, is_leaf, z_type="M2")

