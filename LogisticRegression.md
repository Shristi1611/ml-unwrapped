```python
import numpy as np
from sklearn.datasets import load_breast_cancer

class PureLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        # Maps raw numbers to probabilities between 0 and 1
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Vectorized Gradient Descent Optimization Loop
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Matrix calculus: gradients computed across all dimensions simultaneously
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
        
    def predict(self, X, threshold=0.5):
        # Return crisp binary classes (0 or 1) based on the threshold
        return (self.predict_proba(X) >= threshold).astype(int)

print("--- Day 2: Logistic Regression (Breast Cancer Validation) ---")

# 1. Load raw dataset
cancer_data = load_breast_cancer()
X, y = cancer_data.data, cancer_data.target

# 2. Manual 80/20 train/test split from scratch
np.random.seed(42)  # For reproducible splitting
indices = np.random.permutation(X.shape[0])
split_idx = int(0.8 * X.shape[0])

X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

# 3. Z-score Normalization from scratch to prevent gradient explosion
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# 4. Train the Custom Engine
log_model = PureLogisticRegression(learning_rate=0.1, epochs=1000)
log_model.fit(X_train_scaled, y_train)

# 5. Evaluate Accuracy from scratch
predictions = log_model.predict(X_test_scaled)
accuracy = np.mean(predictions == y_test)

print(f"Dataset Stats -> Samples: {X.shape[0]} | Features: {X.shape[1]}")
print(f"Custom Optimization Engine Accuracy: {accuracy * 100:.2f}%")
```


```python

```


```python

```
