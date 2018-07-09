<div style="background-color: #1f1f1f; padding: 30px;">
<div style="color: #34c232; font-size: 52pt; font-weight: bold">
bonsai 
</div>
<div style="color: #34c232; font-size: 10pt"> 
PROGRAMMABLE DECISION TREE
</div>
</div>

# Bonsai Project

Bonsai is a "programmable" (or "extremely customizable") decision tree. 
Using Bonsai, you can quickly design/build new decision tree algorithms simply by defining two functions:

- `find_split()`
- `is_leaf()`

The intent of this project is to provide a quick testing bed for various decision tree ideas, rather than improve the speed or scalability of decision tree algorithms; although the speed of Bonsai can be comparable with many exisiting implementations.

Many decision trees, including the ones with information gain and gini impurity, are already implemented and available in Bonsai.
Even complex tree models, such as Gradient Boosting, Random Forests and [PaloBoost](), are available in Bonsai.
For more information, please take a look at [our examples](tests/).

## Table of Contents

0. [Installing](#Installing)
0. [Background](#Background)
0. [Design Functions](#Design-Functions)
0. [Code Examples](#Code-Examples)
0. [Glossaries](#Glossaries)
0. [Authors](#Authors)
0. [License](#License)

## Installing

- Required: `numpy` and `cython`
- Run `python setup.py develop`

## Background

Decision tree is a class of recursive data partitioning algorithms. 
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
- `find_split(X, y)`: This function decides which variable/value to split on. For example, C4.5 and ID3 use Information Gain using Shannon Entropy, and CART use the Gini impurity measure to choose a splitting variable. 
- `is_leaf(X, y)`: This function controls when to stop growing the tree. For example, one can grow the tree till its depth is smaller than 4, or its node size is smaller than 30.

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


## Code Examples

### Using Pre-packaged Templates

Take a look at the test scripts under the `tests/` folder. 
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

### Designing Your Own Bonsai

To Be Added (TBA)

## Glossaries 

Deep Dive of Cython Code

- `canvas_dim`: TBA
- `sketch`: TBA
- `canvas`: TBA
- `avc`: TBA
- `setup canvas`: TBA
- `erase canvas`: TBA

## Authors

- Yubin Park

## License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)



