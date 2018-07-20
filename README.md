Bonsai-DT (or Bonsai in short) is a "programmable" decision tree framework. 
Using Bonsai, you can quickly design/build new decision tree algorithms simply by defining two functions:

- `find_split()`
- `is_leaf()`

The intent of this project is to provide a quick testing bed for various decision tree ideas.
The project does not primarily focus on the speed or scalability of decision tree algorithms; 
although the speed of Bonsai can be comparable with many exisiting implementations.

Many decision trees, including the ones with information gain and gini impurity, are already implemented and available in Bonsai.
Even complex tree models, such as Gradient Boosting, Random Forests and [PaloBoost](), are available in the Bonsai templates.
If you want to see the full list of the Bonsai templates, please read [the Model Templates section](#model-templates).


## Contents

0. [Installing](#installing)
0. [Background](#background)
0. [Design Functions](#design-functions)
0. [Code Examples](#code-examples)
0. [Model Templates](#model-templates)
0. [Glossaries](#glossaries)
0. [Authors](#authors)
0. [License](#license)

## Installing

- Required: `numpy` and `cython`
- Run `python setup.py develop`
- NOTE: Bonsai is still under rapid development. Please use this at your own discretion.

## Background

Decision tree is a recursive data partitioning algorithm. 
Some well-known algorithms include [CART (Classification and Regression Trees)](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29), [C4.5](https://en.wikipedia.org/wiki/C4.5_algorithm), [ID3](https://en.wikipedia.org/wiki/ID3_algorithm), and [CHAID](https://en.wikipedia.org/wiki/Chi-square_automatic_interaction_detection). 
Although each decision tree may look differerent, without loss of generality, most trees have the following struture:

```python
def decision_tree(X, y):

    # Stopping criteria
    if is_leaf(X, y):     
        return np.mean(y)

    split_variable, split_value = find_split(X, y)

    # Partition data into two sets
    X_right, X_left, y_right, y_left = partition_data(X, y, 
        split_variable, split_value)  

    # Recursive call of decision_tree()
    return {"split_variable": split_variable,
        "split_value": split_value,
        "right": decision_tree(X_right, y_right), 
        "left": decision_tree(X_left, y_left)}   
```

In the code above, `X` and `y` indicate features and target variable, respectively. 
The algorithm first looks at if it should stop the recursion - `is_leaf()`. 
If not, then it searches a splitting variable and value pair to branch out further - `find_split()`.
Based on the best pair of splitting variable and value, the dataset `(X, y)` is partitioned into two disjoint sets - one with smaller value than the splitting value, and the other with greather than or equal to the splitting value.
We repeat the above process on each partitioned dataset recursively.

The properties of decision tree are primarily determined by the two functions:
- `find_split()`: This function decides which variable/value to split on. For example, C4.5 and ID3 use Information Gain using Shannon Entropy, and CART use the Gini impurity measure to choose a splitting variable. 
- `is_leaf()`: This function controls when to stop growing the tree. For example, one can grow the tree till its depth is smaller than 4, or its node size is smaller than 30.

Many decision tree algorithms out there provide a set of parameters to control the behavior of these two functions. 
Note that, in Bonsai, we ask users to write these **two functions from scratch**.

## Design Functions

### `find_split(avc)`

The argument of the `find_split` is a numpy array named as `avc`. 
The `avc` variable has 10 columns and variable rows.
The name `avc` stands for Attribute-Value Classlabel Group (or AVC-group), which was first introduced in [the Rainforest framework](https://link.springer.com/article/10.1023/A:1009839829793). 
The Bonsai core algorithm "sketches" the statistics of data on a numpy array, named as "canvas", and returns the "AVC-group" (`avc`) to the `find_split` function.
Essentially, the role of the `find_split` function is to examine the distributions of the data, and to find the best attribute and value pair to split the data. 
The column order of `avc` is as follows:

- `avc[:,0]`: AVC indices
- `avc[:,1]`: variable indices i.e. column indices starting from 0
- `avc[:,2]`: split values i.e. samples below this value goes to the left node, and vice versa
- `avc[:,3]`: number of samples at left node
- `avc[:,4]`: sum `y` at left node i.e. `\sum_{left} y`
- `avc[:,5]`: sum of `y^2` at left node i.e. `\sum_{left} y^2`
- `avc[:,6]`: number of samples at right node
- `avc[:,7]`: sum of `y` at right node i.e. `\sum_{right} y`
- `avc[:,8]`: sum of `y^2` at right node i.e. `\sum_{right} y^2`
- `avc[:,9]`: missing value inclusion (0: left node, 1: right node)

Users need to play with this array to define custom splitting criteria. 
 
The return of the function is a dictionary with the following required key value pairs:

- `selected`: the selected row of the `avc` array
- `<additional var>`: any additional variables you want to pass on

For example, if you want to design a splitting criteria that minimizes the weighted sum of variances after the split (e.g. regression tree), you can write `find_split` as follows:

```python
def find_split(avc):
    var_left = avc[:,5]/avc[:,3] - np.square(avc[:,4]/avc[:,3])
    var_right = avc[:,8]/avc[:,6] - np.square(avc[:,7]/avc[:,6])
    varsum = avc[:,3] * var_left  + avc[:,6] * var_right
    min_idx = np.argsort(varsum)[0]
    return {"selected": avc[min_idx,:]}
```
    
### `is_leaf(branch, branch_parent)`

The `is_leaf` function has two arguments: one for the current branch, and the other for the parent branch. Both branches are in the form of Python dictionary. 
By default, these branch variables have the following key value pairs:
- `depth`: the depth of the branch in the tree
- `n_samples`: the number of samples in the branch

Users can add additional key value pairs if necessary.

The return of this function is boolean. If the branch is a leaf, then returns `True`. Otherwise, returns `False`.

For example, if you want to design a stopping criteria that stops at depth 3, you can write `is_leaf` as follows:

```python
def is_leaf(branch, branch_parent):
    if branch["depth"] > 2:
        return True
    else:
        return False
```

### Putting Things Together

With the `find_split` and `is_leaf` implemented, now you can put these together to make your custom decision tree as follows:

```python
from ..core.bonsai import Bonsai
import numpy as np

class MyTree(Bonsai):

    def __init__(self):

        def find_split(avc):
            var_left = avc[:,5]/avc[:,3] - np.square(avc[:,4]/avc[:,3])
            var_right = avc[:,8]/avc[:,6] - np.square(avc[:,7]/avc[:,6])
            varsum = avc[:,3] * var_left  + avc[:,6] * var_right
            min_idx = np.argsort(varsum)[0]
            return {"selected": avc[min_idx,:]}

        def is_leaf(branch, branch_parent):
            if branch["depth"] > 2:
                return True
            else:
                return False

        Bonsai.__init__(self, find_split, is_leaf)
```

Now, you can use `MyTree` just by importing the class in your project. 

## Code Examples

### Using a Model Template

You can use the model templates already built in Bonsai. 
The model templates implemente various decision trees, so you can just import and use them right away.
A regression tree example is as follows:

```python
# Pre-built regression tree using Bonsai
from bonsai.regtree import RegTree 

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_friedman1(n_samples=100000) 
n, m = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize
model = RegTree(max_depth=5),

# Fit
model.fit(X_train, y_train)

# Predict
y_hat = model.predict(X_test)

rmse = np.sqrt(np.mean((y_test - y_hat)**2))
```

Available model templates are listed in [the next section](#model-templates).
Also, take a look at the test scripts under [the tests folder](tests/). 

### Interpreting a Trained Bonsai Tree

In Bonsai, you can easily check the details of a trained tree:

```python
from __future__ import print_function
from bonsai.base.regtree import RegTree
from sklearn.datasets import make_friedman1
import json
X, y = make_friedman1(n_samples=10000) 
model = RegTree(max_depth=1)
model.fit(X, y)
print(json.dumps(model.dump(), indent=2))
```

This script will output the trained tree in the form of an array of leaf nodes. Here is an example of the output:

```json
[
  {
    "is_leaf": true, 
    "eqs": [
      {
        "svar": 3, 
        "missing": 1, 
        "sval": 0.49608099474494854, 
        "op": "<"
      }
    ], 
    "depth": 1, 
    "n_samples": 5031.0, 
    "y": 11.798433612486887, 
  }, 
  {
    "is_leaf": true, 
    "eqs": [
      {
        "svar": 3, 
        "missing": 0, 
        "sval": 0.49608099474494854, 
        "op": ">="
      }
    ], 
    "depth": 1, 
    "n_samples": 4969.0, 
    "y": 16.969889465743922, 
  }
]
```

This is a depth-1 tree, so you have two leaves in the array.
In each leaf, you see `eqs` that stores the logical rules for the leaf.
Also, you see `y`, which indicates the node value or predicted value for the leaf.
All Bonsai-derived trees would have this form of output.

## Model Templates

Here are some Bonsai templates:

- [Regression Tree](bonsai/base/regtree.py) implements the regression tree in CART [(src)](bonsai/base/regtree.py) [(usage)](tests/regtree.py)
- [Alpha Tree](bonsai/base/alphatree.py) implements the [Alpha tree (paper)](https://arxiv.org/abs/1606.05325) [(src)](bonsai/base/alphatree.py) [(usage)](tests/alphatree.py)
- [C45 Tree](bonsai/base/c45tree.py) implements a C4.5-like tree using the alpha tree. The tree uses the information gain for its splitting criterion [(src)](bonsai/base/c45tree.py)
- [Gini Tree](bonsai/base/ginitree.py) implements the classification tree in CART using the alpha tree [(src)](bonsai/base/ginitree.py)
- [XGBoost Base Tree](bonsai/base/xgbtree.py) implements the base learner of [XGBoost (paper)](https://github.com/dmlc/xgboost) [(src)](bonsai/base/xgbtree.p)
- [Friedman Tree](bonsai/base/friedmantree.py) implements the base learner of [the original Gradient Boosting Machine paper](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) [(src)](bonsai/base/friedmantree.py)
- [Gradient Boosting](bonsai/ensemble/gbm.py) implements the [Stochastic Gradient TreeBoost (paper)](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf) [(src)](bonsai/ensemble/gbm.py) [(usage)](tests/gbm.py)
- [Random Forests](bonsai/ensemble/randomforests.py) implements the [Random Forests (paper)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) [(src)](bonsai/ensemble/randomforests.py) [(usage)](tests/randomforests.py)
- [PaloBoost](bonsai/ensemble/paloboost.py) implements the [PaloBoost (paper)]() [(src)](bonsai/ensemble/paloboost.py) [(research)](research/paloboost.ipynb)

## Glossaries 

Variable/function names that are used in the source code:

- `avc`: AVC stands for Attribute-Value Classlabel Group. 
- `canvas`: Canvas is a numpy array that stores AVC. 
- `canvas_dim`: The dimension of Canvas. #rows x #columns
- `setup_canvas`: Setting up a canvas for AVC. 
- `sketch`: Sketching refers to a process of filling AVC values to Canvas. We scan a dataset and construct statistics relevant to AVC, and fill those values to a canvas one by one. We thought the process resembles skteching in paiting - rapid drawing for setting up a blueprint.
- `erase_canvas`: Erase the values in a canvas.

## Authors

- Yubin Park, PhD
- If you want to contribute, please contact Yubin

## License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)



