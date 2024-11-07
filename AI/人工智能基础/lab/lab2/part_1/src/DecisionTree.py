from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np 
import pandas as pd 
from typing import Optional

# metrics
def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

# TreeNode
class TreeNode:
    def __init__(self, feature: int , threshold: Optional[float] = None, values: Optional[dict] = None, label: int = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.values = values if values else None
        self.label = label

    def is_leaf(self) -> bool:
        return self.label is not None
    
    @staticmethod
    def leaf(label:int) -> 'TreeNode':
        return TreeNode(feature=-1, label=label)
    
    def __repr__(self):
        if self.is_leaf():
            return f"Leaf: {self.label}"
        return f"Node: {self.feature}, {self.threshold}"

# model
class DecisionTreeClassifier:
    def __init__(self) -> None:
        self.tree = None

    def _entropy(self, y):
        # y: [n_samples, ]
        # return: entropy
        # entropy = - sum(p_i * log2(p_i))
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p))

    def _information_gain(self, y, y_left, y_right):
        # y: [n_samples, ], y_left: [n_samples_left, ], y_right: [n_samples_right, ]
        # return: information gain
        # information gain = entropy(y) - (n_left/n * entropy(y_left) + n_right/n * entropy(y_right))
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        return self._entropy(y) - (n_left/n * self._entropy(y_left) + n_right/n * self._entropy(y_right))

    def _best_split(self, X: pd.DataFrame, y: np.ndarray) -> tuple:
        # return: best_split_feature, best_split_value
        best_gain = -1
        best_split_feature, best_split_value = None, None

        for feature in X.columns:
            values = np.unique(X[feature])
            if X[feature].dtype.kind in 'bifc':
                values = (values[1:] + values[:-1]) / 2.0
            
            for value in values:
                y_left = y[X[feature] <= value]
                y_right = y[X[feature] > value]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split_feature = feature
                    best_split_value = value
        return best_split_feature, best_split_value

    def Tree_Generate(self, X: pd.DataFrame, y: np.ndarray) -> TreeNode:
        # return: tree
        # tree = {'split_feature': split_feature, 'split_value': split_value, 'left': left_tree, 'right': right_tree}
        if len(np.unique(y)) == 1:
            return TreeNode.leaf(int(y[0]))

        if X.shape[1] == 0:
            return TreeNode.leaf(int(np.argmax(np.bincount(y))))
        
        split_feature, split_value = self._best_split(X, y)
        if split_feature is None:
            return TreeNode.leaf(int(np.argmax(np.bincount(y))))
        
        left_mask = X[split_feature] <= split_value
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        children = {
            'left': self.Tree_Generate(X_left, y_left),
            'right': self.Tree_Generate(X_right, y_right)
        }
        return TreeNode(split_feature, split_value, children, None)
        
        

    def fit(self, X, y):
        # X: [n_samples_train, n_features], 
        # y: [n_samples_train, ],
        # TODO: implement decision tree algorithm to train the model
        self.tree = self.Tree_Generate(X, y)
    
    def tree_predict(self, X: pd.DataFrame, node: TreeNode):
        # X: [n_samples_test, n_features], node: TreeNode
        # return: y: [n_samples_test, ]
        if node.is_leaf():
            return np.array([node.label] * X.shape[0])
        left_mask = X[node.feature] <= node.threshold
        right_mask = ~left_mask
        X_left, X_right = X[left_mask], X[right_mask]
        y = np.zeros(X.shape[0])
        y[left_mask] = self.tree_predict(X_left, node.values['left'])
        y[right_mask] = self.tree_predict(X_right, node.values['right'])
        return y

    def predict(self, X: pd.DataFrame):
        # X: [n_samples_test, n_features],
        # return: y: [n_samples_test, ]
        return self.tree_predict(X, self.tree)
        # TODO:

def load_data(datapath:str='./data/ObesityDataSet_raw_and_data_sinthetic.csv'):
    df = pd.read_csv(datapath)
    continue_features = ['Age', 'Height', 'Weight', ]
    discrete_features = ['Gender', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF', 'TUE', 'CAEC', 'MTRANS']
    
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # encode discrete str to number, eg. male&female to 0&1
    labelencoder = LabelEncoder()
    for col in discrete_features:
        X[col] = labelencoder.fit(X[col]).transform(X[col])
    y = labelencoder.fit(y).fit_transform(y)

    # Scale continue features
    scaler = StandardScaler()
    X[continue_features] = scaler.fit_transform(X[continue_features])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__=="__main__":
    X_train, X_test, y_train, y_test = load_data('./data/ObesityDataSet_raw_and_data_sinthetic.csv')

    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(accuracy(y_test, y_pred))