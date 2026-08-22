# Foundations
> From-scratch implementations of machine learning algorithms: no frameworks, just matrix math and NumPy.

This is the companion repository for **ML Unwrapped**, a 30-day series deconstructing AI architectures down to their mathematical primitives. Each notebook translates the core equations of a classic algorithm directly into clean, vectorized matrix operations.

---

## Philosophy

Most ML tutorials teach you to call `.fit()`. This series teaches you what `.fit()` is actually doing - every gradient, every matrix operation, every assumption that high-level libraries silently make on your behalf.

No scikit-learn. No TensorFlow. No PyTorch. Just NumPy and the math.

---

## Features

- **Pure NumPy architecture** - built entirely on linear algebra primitives, no high-level ML frameworks
- **Vectorized implementations** - matrix operations throughout, no performance-heavy per-sample loops
- **Numerically stable** - edge cases handled correctly (stable sigmoid, stable softmax, lstsq over explicit inverse)
- **Real-world validation** - every algorithm tested against standard datasets, not toy examples
- **Self-contained files** - each `.md` includes the math breakdown, implementation, and validation pipeline

---

## Algorithms

| Day | Algorithm | Key Concept | Dataset | Accuracy |
|-----|-----------|-------------|---------|----------|
| 01 | [Linear Regression](01_LinearRegression.md) | Normal Equation - closed-form exact solution | California Housing (20,640 samples) | - |
| 02 | [Logistic Regression](02_LogisticRegression.md) | Sigmoid + gradient descent - binary classification | Breast Cancer Wisconsin (569 samples) | 97.37% |
| 03 | [Softmax Regression](03_MultiClassRegression.md) | Softmax + one-hot encoding - multi-class classification | Iris (150 samples, 3 classes) | 96.67% |
| 04 | [Gaussian Naive Bayes](04_NaiveBayes.md) | Bayes theorem - probabilistic classification, no training loop | UCI Forensic Glass (214 samples, 6 classes) | 51.16% |
| 05 | [K-Nearest Neighbors](05_KNN.md) | Euclidean distance - instance-based, no parameters | UCI Forensic Glass (214 samples, 6 classes) | 62.79% |
| 06 | [Decision Tree](06_DecisionTree.md) | Entropy + information gain - recursive binary splitting | UCI Forensic Glass (214 samples, 6 classes) | 67.44% |
| ... | More to come | Neural Networks, SVMs, Random Forests, and more | - | - |

---

## Project Structure

```text
Foundations/
├── 01_LinearRegression.md      # Day 01 - Normal Equation, California Housing dataset
├── 02_LogisticRegression.md    # Day 02 - Sigmoid, gradient descent, Breast Cancer dataset
├── 03_MultiClassRegression.md  # Day 03 - Softmax, one-hot encoding, Iris dataset
├── 04_NaiveBayes.md            # Day 04 - Bayes theorem, UCI Forensic Glass dataset
├── 05_KNN.md                   # Day 05 - Euclidean distance, UCI Forensic Glass dataset
├── 06_DecisionTree.md          # Day 06 - Entropy, information gain, UCI Forensic Glass dataset
└── ...                         # Days 07-30 in progress
```

---

## The Math at a Glance

**Day 01 - Linear Regression (Normal Equation)**
```
b = (XtX)^-1 Xty
```
Closed-form. No iterations. Finds the exact optimal weights in one matrix operation.

**Day 02 - Logistic Regression (Gradient Descent)**
```
s(z) = 1 / (1 + e^-z)
dw = (1/n) Xt(y_hat - y)
```
No closed-form solution exists. Gradient descent iterates to convergence.

**Day 03 - Softmax Regression (Multi-Class)**
```
s(zi) = e^zi / sum(e^zj)
```
Generalises logistic regression to K classes. Outputs a probability distribution that sums to 1.

**Day 04 - Gaussian Naive Bayes (Probabilistic)**
```
P(y | x) ~ P(x | y) * P(y)
log posterior = log(prior) + sum(log P(xi | y))
```
No gradient descent. Fits a Gaussian per feature per class, predicts via Bayes theorem.

**Day 05 - K-Nearest Neighbors (Instance-Based)**
```
d(x, y) = sqrt(sum((xi - yi)^2))
```
No training. Stores the dataset, predicts by majority vote across k nearest neighbors.

**Day 06 - Decision Tree (Entropy Engine)**
```
IG = H(parent) - sum(w * H(child))
H(y) = -sum(p * log2(p))
```
Recursively splits on the feature and threshold that maximises information gain.

---

## Requirements

```bash
pip install numpy jupyter scikit-learn pandas
```

> `scikit-learn` is used only for loading standard datasets, never for model training or evaluation.

---

## Part of ML Unwrapped

This repository is updated daily as part of the **ML Unwrapped** 30-day series. Follow along on [LinkedIn](https://www.linkedin.com/in/shristi-sinha-) for the breakdown behind each build.

30 days. 30 algorithms. All math, no magic.
