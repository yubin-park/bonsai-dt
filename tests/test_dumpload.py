from bonsai.base.regtree import RegTree
from sklearn.datasets import make_friedman1
import pickle
import numpy as np
import unittest

class DumpLoadTestCase(unittest.TestCase):

    def test_dumpload(self):

        n_samples = 10000
        max_depth = 3

        X, y = make_friedman1(n_samples=n_samples) 
        model = RegTree(max_depth=max_depth)
        model.fit(X, y)

        z = model.predict(X)
        mse_pre = np.mean((z - y)**2)

        s = pickle.dumps(model.dump())
        model.load(pickle.loads(s))
        mse_post = np.mean((model.predict(X) - y)**2)

        self.assertAlmostEqual(mse_pre, mse_post)

if __name__=="__main__":

    unittest.main()


