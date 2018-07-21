**Bonsai-DT** (or **Bonsai** in short form) is a *programmable* decision tree framework written in Python (and a little bit of Cython). 
Using Bonsai, you can quickly design/build/customize new decision tree algorithms simply by writing these two functions:

- `find_split()`
- `is_leaf()`

The intent of this project is to provide a quick testing bed for various decision tree ideas.
Although the speed of Bonsai is comparable to other implementations, we note that the speed and scalability of the framework are not the primary focus.

Many decision trees, such as C4.5 and CART, are already implemented and available in the Bonsai templates.
Even ensemble models, such as Gradient Boosting, Random Forests and [PaloBoost](), are readily available.
For the full list of the Bonsai templates, please see [the Model Templates section](#model-templates).

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
Although these decision trees may look very differerent from outside, 
without loss of generality, the inner-workings of the most trees can be written as follows:

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

In the code above, `X` and `y` represent feature matrix and target vector, respectively. 
The algorithm first looks at if it should stop the recursion - `is_leaf()`. 
If not, then it searches a splitting variable and value pair to branch out further - `find_split()`.
Based on the best pair of splitting variable and value, the dataset `(X, y)` is partitioned into two disjoint sets - one with smaller value than the splitting value, and the other with greather than or equal to the splitting value.
Finally, the algorithm recursively repeats the above process on each partitioned dataset.

As can be seen, the behavior of a decision tree is primarily governed by the two functions:
- `find_split()` decides which variable/value to split on. For example, C4.5 and ID3 use Information Gain using Shannon Entropy, and CART uses the Gini impurity measure (for classification) and the minimum variance criterion (for regression) to choose a splitting variable. 
- `is_leaf()` controls when to stop growing the tree. For example, a tree can stop growing if its depth is greater than 4 or its node size is smaller than 30.

Many decision tree algorithms implement these functions internally, and expose only a set of parameters through their interfaces. 
This approach is definitely better for users who want to try the off-the-shelf decision trees.
However, if we want to design *a new trees* that does not exist anywhere, 
we may need to edit the underlying source code, which can be quite complex and time-consuming.
Is there a better way?
That's why we developed Bonsai.

## Design Functions

Rather than a set of parameters, Bonsai takes two functions as its arguments.
In this section, we illustrate the details of these functions, such as arguments to pass on, output formats, and some example implementations.

### `find_split(avc)`

The `find_split` function has only one argument `avc`.
The `avc` variable is a numpy array that has 10 columns and variable rows.
The name `avc` stands for Attribute-Value Classlabel Group (or AVC-group), which was first introduced in [the Rainforest framework](https://link.springer.com/article/10.1023/A:1009839829793). 


Here is how Bonsai works in a nutshell.
Whenever we call `find_split`, Bonsai scans the data, and updating a couple of online statistics such as mean, second moment, count, etc.
We call this process as Bonsai "sketches" the stats on `canvas`, where canvas represents just an empty numpy array.
When the sketch is done, the empty numpy array, `canvas`, is filled with all relevant statistics for defining splitting criteria. 
Just to be clear, we call the `canvas` after sketch as `avc`. 
The column order of `avc` is as follows:

- `avc[:,0]`: AVC indices
- `avc[:,1]`: variable indices i.e. column indices starting from 0
- `avc[:,2]`: split values i.e. samples below this value goes to the left node, and vice versa
- `avc[:,3]`: number of samples at (hypothetical) left node
- `avc[:,4]`: sum `y` at left node i.e. `\sum_{left} y`
- `avc[:,5]`: sum of `y^2` at left node i.e. `\sum_{left} y^2`
- `avc[:,6]`: number of samples at (hypothetical) right node
- `avc[:,7]`: sum of `y` at right node i.e. `\sum_{right} y`
- `avc[:,8]`: sum of `y^2` at right node i.e. `\sum_{right} y^2`
- `avc[:,9]`: missing value inclusion (0: left node, 1: right node)

Essentially, the `find_split` function is given with the joint distribution information in the form of AVC-group.
With this information, the function needs to define what properties of the distributions should paly roles in finding the best splitting variable and value. 
 
The return of the function is a Python dictionary with the following required key value pairs:

- `selected`: the selected row of the `avc` array
- `<additional var>`: any additional variables you want to pass on

For example, if you want to design a splitting criteria that minimizes the weighted sum of variances after the split (e.g. regression tree in CART), you can write `find_split` as follows:

```python
def find_split(avc):
    var_left = avc[:,5]/avc[:,3] - np.square(avc[:,4]/avc[:,3])
    var_right = avc[:,8]/avc[:,6] - np.square(avc[:,7]/avc[:,6])
    varsum = avc[:,3] * var_left  + avc[:,6] * var_right
    min_idx = np.argsort(varsum)[0]
    return {"selected": avc[min_idx,:]}
```
    
### `is_leaf(branch, branch_parent)`

The `is_leaf` function takes two arguments: one for the current branch, and the other for the parent branch. 
Both variables are Python dictionaries. 
By default, these branch variables have the following key value pairs, but can have additional key value pairs if users defined `<additional var>` in `find_split`:
- `depth`: the depth of the branch in the tree
- `n_samples`: the number of samples in the branch

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

With the `find_split` and `is_leaf` written, now you can put these together to make your own decision tree as follows:

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

We provide several examples of using Bonsai.

### Using a Model Template

Probably the easiest and fastest way to use Bonsai is using the model templeates in Bonsai.
The model templates are decision trees that are already built in Bonsai, which include regression tree, C4.5, Alpha-Tree, etc.
For example, if you want to use a regression tree example:

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

Available model templates are listed in [the Model Templates section](#model-templates).
Also, take a look at the test scripts under [the tests folder](https://github.com/yubin-park/bonsai-dt/tree/master/tests) to see how to use these templates. 

### Running a Test Script

The best to way to get used to Bonsai is try running the test scripts.
To run the test scripts, if you just go to the tests folder, and run a Python script in the folder.

```
$ cd tests
$ python paloboost.py 

# Test Regression
-----------------------------------------------------
 model_name     train_time     predict_time   rmse   
-----------------------------------------------------
 baseline       -              -              6.99054
 gbm            3.24681 sec    0.32504 sec    5.76867
 palobst        4.15213 sec    0.28674 sec    5.14846
 sklearn        6.36740 sec    0.07248 sec    5.68470
-----------------------------------------------------
```

This test script tests the PaloBoost algorithm using the Friedman's simulated data (regression task). 
As can be seen, the test script prints the training time, predictio tiem, and Root Mean Squared Errors (RMSE) measured on hold-out data.

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

As this is a depth-1 tree, we only see two leaves in the array.
In each leaf, you see `eqs` that stores the logical rules for the leaf, and `y` that indicates the node value or predicted value for the leaf.
All Bonsai-derived trees would have this form of output.

## Model Templates

Here is the list of Bonsai templates:

- *class* **RegTree()** implements the regression tree in CART. [(src)](https://github.com/yubin-park/bonsai-dt/blob/master/bonsai/base/regtree.py) [(usage)](https://github.com/yubin-park/bonsai-dt/blob/master/tests/regtree.py)
- *class* **AlphaTree**: implements the alpha tree in ["ACDC: alpha-Carving Decision Chain for Risk Stratification"](https://arxiv.org/abs/1606.05325). [(src)](https://github.com/yubin-park/bonsai-dt/blob/master/bonsai/base/alphatree.py) [(usage)](https://github.com/yubin-park/bonsai-dt/blob/master/tests/alphatree.py)
- *class* **C45Tree()** implements a C4.5-like tree using the alpha tree. The tree uses the information gain for its splitting criterion. [(src)](https://github.com/yubin-park/bonsai-dt/blob/master/bonsai/base/c45tree.py)
- *class* **GiniTree()** implements the classification tree in CART using the alpha tree. [(src)](https://github.com/yubin-park/bonsai-dt/blob/master/bonsai/base/ginitree.py)
- *class* **XGBTree()** implements the base learner of [XGBoost](https://github.com/dmlc/xgboost). [(src)](https://github.com/yubin-park/bonsai-dt/blob/master/bonsai/base/xgbtree.py)
- *class* **FriedmanTree()** implements the base learner in ["Greedy Function Approximation: A Gradient Boosting Machine"](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf). [(src)](https://github.com/yubin-park/bonsai-dt/blob/master/bonsai/base/friedmantree.py)
- *class* **GBM()** implements the Stochastic Gradient TreeBoost that appeared in ["Stochastic Gradient Boosting"](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf). [(src)](https://github.com/yubin-park/bonsai-dt/blob/master/bonsai/ensemble/gbm.py) [(usage)](https://github.com/yubin-park/bonsai-dt/blob/master/tests/gbm.py)
- *class* **RandomForests()** implements the Random Forests model in ["Random Forests"](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf). [(src)](https://github.com/yubin-park/bonsai-dt/blob/master/bonsai/ensemble/randomforests.py) [(usage)](https://github.com/yubin-park/bonsai-dt/blob/master/tests/randomforests.py)
- *class* **PaloBoost()** implements PaloBoost in ["PaloBoost: An Overfitting-robust TreeBoost with Out-of-Bag Sample Regularization Techniques"](). [(src)](https://github.com/yubin-park/bonsai-dt/blob/master/bonsai/ensemble/paloboost.py) [(usage)](https://github.com/yubin-park/bonsai-dt/blob/master/tests/paloboost.py) [(research)](https://github.com/yubin-park/bonsai-dt/blob/master/research/paloboost.ipynb)

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



