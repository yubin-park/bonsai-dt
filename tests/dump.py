from bonsai.base.regtree import RegTree
from sklearn.datasets import make_friedman1
import pickle
import numpy as np
import sys

X, y = make_friedman1(n_samples=10000) 
model = RegTree(max_depth=1)
model.fit(X, y)

s = pickle.dumps(model.dump())
print("Size of the model: {}".format(sys.getsizeof(s)))

model.load(pickle.loads(s))
mse = np.mean((model.predict(X) - y)**2)
print("Unpickled model's R2: {}".format(mse/np.var(y)))


