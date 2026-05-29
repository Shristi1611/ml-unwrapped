# Foundations

From-scratch implementations of fundamental machine learning algorithms. Built using raw matrix operations and vectorization techniques to explore the mathematical mechanics behind modern prediction engines.

## Overview

This repository serves as a code-first exploration of machine learning algorithms built completely from first principles using NumPy. By bypassing high-level library abstractions (like scikit-learn or PyTorch), this project focuses on translating analytical equations and optimization calculus directly into clean, efficient, vectorized matrix operations.

## Features

* **Pure NumPy Architecture:** No external high-level ML frameworks; built entirely on linear algebra primitives.
* **Vectorized Implementations:** Avoids performance-heavy loops by utilizing matrix operations.
* **Real-World Benchmarking:** Every architecture is validated against standard datasets to ensure mathematical precision.

---

## Project Structure

```text
├── LinearRegression/
│   ├── pure_regression.py     # Custom OLS engine class from scratch
│   └── housing_pipeline.py    # California Housing dataset training script
└── README.md
