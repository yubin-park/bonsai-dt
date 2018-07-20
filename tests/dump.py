from __future__ import print_function
from bonsai.base.regtree import RegTree
from sklearn.datasets import make_friedman1
import json

X, y = make_friedman1(n_samples=10000) 
model = RegTree(max_depth=1)
model.fit(X, y)
print(json.dumps(model.dump(), indent=2))

