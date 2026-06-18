```python
import numpy as np
import pandas as pd


# =====================================================================
# DAY 4: GAUSSIAN NAIVE BAYES (FORENSIC GLASS CLASSIFIER) FROM SCRATCH
# =====================================================================
class PureGaussianNB:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize arrays to store mean, variance, and prior probability for each class
        self.means = np.zeros((n_classes, n_features))
        self.vars = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        # Calculate parameters for each class independently
        for idx, c in enumerate(self.classes):
            # Isolate only the samples belonging to glass type 'c'
            X_c = X[y == c]

            # Calculate and store statistical profiles
            self.means[idx, :] = np.mean(X_c, axis=0)
            self.vars[idx, :] = np.var(X_c, axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def _calculate_likelihood(self, class_idx, x):
        # Gaussian Probability Density Function (PDF)
        mean = self.means[class_idx]
        var = self.vars[class_idx]

        # Epsilon prevents division by zero on near-zero variance features
        eps = 1e-9
        numerator = np.exp(-((x - mean) ** 2) / (2 * (var + eps)))
        denominator = np.sqrt(2 * np.pi * (var + eps))

        return numerator / denominator

    def predict(self, X):
        predictions = [self._predict_single_sample(x) for x in X]
        return np.array(predictions)

    def predict_proba(self, X):
        all_posteriors = []
        for x in X:
            posteriors = []
            for idx in range(len(self.classes)):
                prior = np.log(self.priors[idx])
                likelihood = np.sum(np.log(self._calculate_likelihood(idx, x) + 1e-9))
                posteriors.append(prior + likelihood)
            # Normalise log posteriors to probabilities via softmax trick
            posteriors = np.array(posteriors)
            posteriors -= np.max(posteriors)
            exp_p = np.exp(posteriors)
            all_posteriors.append(exp_p / exp_p.sum())
        return np.array(all_posteriors)

    def _predict_single_sample(self, x):
        posteriors = []

        # Calculate posterior probability for each glass class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])

            # Under the Naive independence assumption: log P(X|y) = sum of log likelihoods
            likelihood = np.sum(np.log(self._calculate_likelihood(idx, x) + 1e-9))

            # Posterior score = log(Prior) + log(Likelihood)
            posteriors.append(prior + likelihood)

        # Glass type with highest posterior wins
        return self.classes[np.argmax(posteriors)]


# =====================================================================
# DATASET PIPELINE & FORENSIC VALIDATION
# =====================================================================
print("--- Day 4: Gaussian Naive Bayes (Forensic Glass Classification) ---")

# 1. Fetch data directly via URL — self-contained, no local file needed
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
column_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
df = pd.read_csv(url, names=column_names)

# Extract features (drop ID and Type) and target
X = df.drop(columns=["Id", "Type"]).to_numpy()
y = df["Type"].to_numpy()

# 2. Manual 80/20 train/test split from scratch
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

# 3. Train the probability engine
# No scaling needed — Naive Bayes models each feature's distribution independently
model = PureGaussianNB()
model.fit(X_train, y_train)

# 4. Evaluate overall accuracy
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)

print(f"Total Glass Samples : {X.shape[0]} | Features: {X.shape[1]} | Distinct Glass Types: {len(np.unique(y))}")
print(f"Custom Forensic Naive Bayes Engine Accuracy: {accuracy * 100:.2f}%")
print("---")

# 5. Per-class accuracy breakdown — note: UCI glass dataset skips class 4
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
for c in model.classes:
    mask = y_test == c
    if mask.sum() > 0:
        class_acc = np.mean(predictions[mask] == y_test[mask])
        print(f"   Type {c} ({glass_types.get(c, 'Unknown'):<20}) : {class_acc * 100:.1f}%")

print("---")

# 6. Single sample prediction with full class probability distribution
sample = X_test[0].reshape(1, -1)
probas = model.predict_proba(sample)[0]
pred = model.predict(sample)[0]
print(f"Sample Prediction: Type {pred} — {glass_types.get(pred, 'Unknown')}")
print("Class Probabilities:")
for c, prob in zip(model.classes, probas):
    print(f"   Type {c} ({glass_types.get(c, 'Unknown'):<20}) : {prob * 100:.1f}%")
```

    --- Day 4: Gaussian Naive Bayes (Forensic Glass Classification) ---
    Total Glass Samples : 214 | Features: 9 | Distinct Glass Types: 6
    Custom Forensic Naive Bayes Engine Accuracy: 51.16%
    ---
    Per-Class Accuracy Breakdown:
       Type 1 (Building Float      ) : 66.7%
       Type 2 (Building Non-Float  ) : 33.3%
       Type 3 (Vehicle Float       ) : 25.0%
       Type 5 (Vehicle Non-Float   ) : 50.0%
       Type 6 (Container           ) : 100.0%
       Type 7 (Tableware           ) : 62.5%
    ---
    Sample Prediction: Type 1 — Building Float
    Class Probabilities:
       Type 1 (Building Float      ) : 81.8%
       Type 2 (Building Non-Float  ) : 15.9%
       Type 3 (Vehicle Float       ) : 2.3%
       Type 5 (Vehicle Non-Float   ) : 0.0%
       Type 6 (Container           ) : 0.0%
       Type 7 (Tableware           ) : 0.0%
    


```python

```
