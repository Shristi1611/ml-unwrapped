```python
import numpy as np
import pandas as pd
from collections import Counter

class PureKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict_single_sample(x) for x in X_test]
        return np.array(predictions)

    def predict_proba(self, X_test):
        all_probas = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            counts = Counter(k_nearest_labels)
            all_probas.append({c: counts.get(c, 0) / self.k for c in np.unique(self.y_train)})
        return all_probas

    def _predict_single_sample(self, x_test):
        distances = np.sqrt(np.sum((self.X_train - x_test) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

print("--- Day 5: K-Nearest Neighbors (Forensic Glass Classification) ---")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
column_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
df = pd.read_csv(url, names=column_names)
X = df.drop(columns=["Id", "Type"]).to_numpy()
y = df["Type"].to_numpy()

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

np.random.seed(42)
indices = np.random.permutation(X_scaled.shape[0])
split_idx = int(0.8 * X_scaled.shape[0])
X_train, X_test = X_scaled[indices[:split_idx]], X_scaled[indices[split_idx:]]
y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

model = PureKNN(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)

print(f"Dataset Stats -> Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {len(np.unique(y))}")
print(f"Custom KNN Engine (k=3) Test Accuracy: {accuracy * 100:.2f}%")
print("---")

print("K-Sweep (Accuracy vs K):")
for k in [1, 3, 5, 7, 9, 11]:
    m = PureKNN(k=k)
    m.fit(X_train, y_train)
    acc = np.mean(m.predict(X_test) == y_test)
    print(f"   k={k:>2} : {acc * 100:.2f}%")

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
for c in np.unique(y_test):
    mask = y_test == c
    if mask.sum() > 0:
        class_acc = np.mean(predictions[mask] == y_test[mask])
        print(f"   Type {c} ({glass_types.get(c, 'Unknown'):<20}) : {class_acc * 100:.1f}%")

print("---")

sample = X_test[0].reshape(1, -1)
pred = model.predict(sample)[0]
probas = model.predict_proba(sample)[0]
print(f"Sample Prediction: Type {pred} - {glass_types.get(pred, 'Unknown')}")
print("Neighbor Vote Share:")
for c, share in sorted(probas.items()):
    if share > 0:
        print(f"   Type {c} ({glass_types.get(c, 'Unknown'):<20}) : {share * 100:.1f}% of votes")
```

    --- Day 5: K-Nearest Neighbors (Forensic Glass Classification) ---
    Dataset Stats -> Samples: 214 | Features: 9 | Classes: 6
    Custom KNN Engine (k=3) Test Accuracy: 62.79%
    ---
    K-Sweep (Accuracy vs K):
       k= 1 : 67.44%
       k= 3 : 62.79%
       k= 5 : 60.47%
       k= 7 : 55.81%
       k= 9 : 58.14%
       k=11 : 58.14%
    ---
    Per-Class Accuracy Breakdown:
       Type 1 (Building Float      ) : 91.7%
       Type 2 (Building Non-Float  ) : 46.7%
       Type 3 (Vehicle Float       ) : 25.0%
       Type 5 (Vehicle Non-Float   ) : 50.0%
       Type 6 (Container           ) : 100.0%
       Type 7 (Tableware           ) : 62.5%
    ---
    Sample Prediction: Type 1 - Building Float
    Neighbor Vote Share:
       Type 1 (Building Float      ) : 66.7% of votes
       Type 2 (Building Non-Float  ) : 33.3% of votes
    


```python

```
