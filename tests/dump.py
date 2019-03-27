from __future__ import print_function
from bonsai.base.regtree import RegTree
from sklearn.datasets import make_friedman1
import json
import numpy as np
X, y = make_friedman1(n_samples=10000) 
model = RegTree(max_depth=1)
model.fit(X, y)

out = model.dump(compact=True)
print(json.dumps(out, indent=2))


model.load(out)

mse = np.mean((model.predict(X) - y)**2)
print(mse)
print(np.var(y))


