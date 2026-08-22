```python
import numpy as np
import pandas as pd


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class PureDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return Node(value=self._most_common_label(y))

        best_feat, best_thresh = self._best_split(X, y, n_samples, n_features)

        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh

        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left_child, right_child)

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))

    def _best_split(self, X, y, n_samples, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thresh in thresholds:
                gain = self._information_gain(y, X_column, thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs = X_column <= threshold
        right_idxs = X_column > threshold

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


print("--- Day 6: Decision Tree Classifier (Entropy Engine Validation) ---")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
column_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
df = pd.read_csv(url, names=column_names)

X = df.drop(columns=["Id", "Type"]).to_numpy()
y_raw = df["Type"].to_numpy()

classes = np.unique(y_raw)
label_map = {c: i for i, c in enumerate(classes)}
inverse_map = {i: c for c, i in label_map.items()}
y = np.array([label_map[c] for c in y_raw])

np.random.seed(42)
indices = np.random.permutation(X.shape[0])
split_idx = int(0.8 * X.shape[0])

X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
y_test_raw = y_raw[indices[split_idx:]]

model = PureDecisionTree(max_depth=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions_raw = np.array([inverse_map[p] for p in predictions])
accuracy = np.mean(predictions_raw == y_test_raw)

print(f"Dataset Stats -> Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {len(classes)}")
print(f"Custom Decision Tree (max_depth=5) Test Accuracy: {accuracy * 100:.2f}%")
print("---")

print("Depth Sweep (Accuracy vs Max Depth):")
for depth in [1, 2, 3, 4, 5, 6, 7]:
    m = PureDecisionTree(max_depth=depth)
    m.fit(X_train, y_train)
    preds = np.array([inverse_map[p] for p in m.predict(X_test)])
    acc = np.mean(preds == y_test_raw)
    print(f"   depth={depth} : {acc * 100:.2f}%")

print("---")

glass_types = {
    1: "Building Float",
    2: "Building Non-Float",
    3: "Vehicle Float",
    5: "Vehicle Non-Float",
    6: "Container",
    7: "Tableware",
    9: "Headlamp"
}

print("Per-Class Accuracy Breakdown:")
for c in classes:
    mask = y_test_raw == c
    if mask.sum() > 0:
        class_acc = np.mean(predictions_raw[mask] == y_test_raw[mask])
        print(f"   Type {c} ({glass_types.get(c, 'Unknown'):<20}) : {class_acc * 100:.1f}%")

print("---")

sample = X_test[0].reshape(1, -1)
pred_raw = inverse_map[model.predict(sample)[0]]
actual_raw = y_test_raw[0]
print(f"Sample Prediction : Type {pred_raw} - {glass_types.get(pred_raw, 'Unknown')}")
print(f"Actual Label      : Type {actual_raw} - {glass_types.get(actual_raw, 'Unknown')}")
```

    --- Day 6: Decision Tree Classifier (Entropy Engine Validation) ---
    Dataset Stats -> Samples: 214 | Features: 9 | Classes: 6
    Custom Decision Tree (max_depth=5) Test Accuracy: 67.44%
    ---
    Depth Sweep (Accuracy vs Max Depth):
       depth=1 : 44.19%
       depth=2 : 55.81%
       depth=3 : 62.79%
       depth=4 : 67.44%
       depth=5 : 67.44%
       depth=6 : 69.77%
       depth=7 : 69.77%
    ---
    Per-Class Accuracy Breakdown:
       Type 1 (Building Float      ) : 75.0%
       Type 2 (Building Non-Float  ) : 66.7%
       Type 3 (Vehicle Float       ) : 25.0%
       Type 5 (Vehicle Non-Float   ) : 100.0%
       Type 6 (Container           ) : 100.0%
       Type 7 (Tableware           ) : 62.5%
    ---
    Sample Prediction : Type 2 - Building Non-Float
    Actual Label      : Type 2 - Building Non-Float
    


```python

```
