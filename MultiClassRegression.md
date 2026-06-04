```python
import numpy as np
from sklearn.datasets import load_iris


class PureSoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _softmax(self, z):
        # Numerical stability trick: subtract max to prevent np.exp() overflow
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, n_classes, verbose=False):
        self.loss_history = []
        n_samples, n_features = X.shape

        # Initialize weights matrix (features x classes) and bias vector (1 x classes)
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        # One-hot encode target vector y from scratch
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1.0

        # Vectorized Gradient Descent Loop
        for epoch in range(self.epochs):
            # 1. Compute raw scores for all classes: shape (n_samples, n_classes)
            linear_model = np.dot(X, self.weights) + self.bias

            # 2. Convert raw scores to probabilities using Softmax
            y_predicted = self._softmax(linear_model)

            # 3. Compute gradients across all dimensions & classes simultaneously
            error = y_predicted - y_one_hot
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error, axis=0, keepdims=True)

            # 4. Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Cross-entropy loss — 1e-9 epsilon prevents log(0)
            loss = -np.mean(np.sum(y_one_hot * np.log(y_predicted + 1e-9), axis=1))
            self.loss_history.append(loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:>4} | Loss: {loss:.4f}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._softmax(linear_model)

    def predict(self, X):
        # The class with the highest probability wins
        return np.argmax(self.predict_proba(X), axis=1)

# DATASET PIPELINE & VALIDATION

print("--- Day 3: Softmax Regression (Iris Multi-Class Validation) ---")

# 1. Load multi-class dataset
iris = load_iris()
X, y = iris.data, iris.target
n_classes = len(np.unique(y))  # 3 distinct classes

# 2. Manual 80/20 train/test split from scratch
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

# 3. Z-score normalization from scratch
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# 4. Train the custom multi-class engine
model = PureSoftmaxRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train_scaled, y_train, n_classes=n_classes, verbose=True)

# 5. Evaluate overall accuracy from scratch
predictions = model.predict(X_test_scaled)
accuracy = np.mean(predictions == y_test)

print("---")
print(f"Dataset Stats -> Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {n_classes}")
print(f"Custom Softmax Engine Test Accuracy: {accuracy * 100:.2f}%")
print(f"Final Loss: {model.loss_history[-1]:.4f}")

# 6. Per-class accuracy breakdown
print("\nPer-Class Accuracy Breakdown:")
for i, name in enumerate(iris.target_names):
    mask = y_test == i
    class_acc = np.mean(predictions[mask] == y_test[mask])
    print(f"   {name:<12} : {class_acc * 100:.1f}%")

print("---")

# 7. Single sample prediction — shows all 3 class probabilities simultaneously
sample = X_test_scaled[0].reshape(1, -1)
probas = model.predict_proba(sample)[0]
pred = model.predict(sample)[0]
print(f"Sample Prediction: {iris.target_names[pred]}")
print("Class Probabilities:")
for name, prob in zip(iris.target_names, probas):
    print(f"   {name:<12} : {prob * 100:.1f}%")
```

    --- Day 3: Softmax Regression (Iris Multi-Class Validation) ---
    Epoch    0 | Loss: 1.0986
    Epoch  100 | Loss: 0.3226
    Epoch  200 | Loss: 0.2599
    Epoch  300 | Loss: 0.2226
    Epoch  400 | Loss: 0.1970
    Epoch  500 | Loss: 0.1783
    Epoch  600 | Loss: 0.1640
    Epoch  700 | Loss: 0.1528
    Epoch  800 | Loss: 0.1438
    Epoch  900 | Loss: 0.1363
    ---
    Dataset Stats -> Samples: 150 | Features: 4 | Classes: 3
    Custom Softmax Engine Test Accuracy: 96.67%
    Final Loss: 0.1301
    
    Per-Class Accuracy Breakdown:
       setosa       : 100.0%
       versicolor   : 100.0%
       virginica    : 91.7%
    ---
    Sample Prediction: versicolor
    Class Probabilities:
       setosa       : 0.7%
       versicolor   : 82.6%
       virginica    : 16.6%
    


```python

```
