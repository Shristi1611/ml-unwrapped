```python
import numpy as np
from sklearn.datasets import fetch_california_housing


class PureLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        # Prepend bias column of ones — vectorization trick
        ones = np.ones((X.shape[0], 1))
        X_matrix = np.hstack((ones, X))

        # Solve the Normal Equation using lstsq (numerically stable — avoids explicit inverse)
        # Equivalent to: beta = (XᵀX)⁻¹ Xᵀy, but without computing the inverse directly
        beta, _, _, _ = np.linalg.lstsq(X_matrix, y, rcond=None)

        # Extract bias and weights from solution vector
        self.bias = float(beta[0])
        self.weights = beta[1:].flatten()

    def predict(self, X):
        X = np.array(X)

        # Prepend bias column — mirrors fit() structure exactly
        ones = np.ones((X.shape[0], 1))
        X_matrix = np.hstack((ones, X))

        # Reconstruct full parameter vector and apply in one matrix op
        beta = np.concatenate([[self.bias], self.weights])
        return (X_matrix @ beta).flatten()


if __name__ == "__main__":
    print("Fetching California Housing dataset...")
    housing_data = fetch_california_housing()

    X = housing_data.data
    y = housing_data.target  # Values scaled in $100,000s

    print(f"Dataset successfully loaded! Shape of X: {X.shape}")
    print("---")

    model = PureLinearRegression()
    print("Training the engine on real data using matrix algebra...")
    model.fit(X, y)
    print("Complete!")
    print("---")

    # Inspect what the math extracted
    print("INTERCEPT (Calculated Bias):")
    print(f"   {model.bias:.4f}")
    print("\nCOEFFICIENTS (Calculated Weights for each feature):")
    for name, weight in zip(housing_data.feature_names, model.weights):
        print(f"   {name:<12} : {weight: .4f}")
    print("---")

    # Prediction test
    # Order = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    sample_block = np.array([[4.5, 20.0, 5.0, 1.0, 1000.0, 3.0, 34.0, -118.0]])

    predicted_price = model.predict(sample_block)
    print(f"Prediction for our test property block: ${predicted_price[0] * 100000:,.2f}")
```

    Fetching California Housing dataset...
    Dataset successfully loaded! Shape of X: (20640, 8)
    ---
    Training the engine on real data using matrix algebra...
    Complete!
    ---
    INTERCEPT (Calculated Bias):
       -36.9419
    
    COEFFICIENTS (Calculated Weights for each feature):
       MedInc       :  0.4367
       HouseAge     :  0.0094
       AveRooms     : -0.1073
       AveBedrms    :  0.6451
       Population   : -0.0000
       AveOccup     : -0.0038
       Latitude     : -0.4213
       Longitude    : -0.4345
    ---
    Prediction for our test property block: $225,296.89
    

    C:\Users\shris\AppData\Local\Temp\ipykernel_29180\1033390831.py:23: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
      self.bias = float(beta[0])
    


```python

```
