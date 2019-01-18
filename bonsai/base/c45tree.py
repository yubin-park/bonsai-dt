"""
This class implements C45 by inheriting the alpha tree (alpha=1).
"""

# Authors: Yubin Park <yubin.park@gmail.com>
# License: Apache License 2.0

from bonsai.base.alphatree import AlphaTree
import numpy as np

class C45Tree(AlphaTree):

    def __init__(self, 
                max_depth=5, 
                min_samples_split=2,
                min_samples_leaf=1):

        AlphaTree.__init__(self, 
                        alpha=1.0,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf)


