<h1 align="center">Solutions</h1>

## ❓ Missing values

1. `np.NaN`
2. `df.replace(9999, np.NaN)`
3. `df.isnull().sum()` or `df.isna().sum()`
4. `df.isnull().sum() / len(df)` or `df.isna().sum() / len(df)`
5. `df.dropna(axis=0)` or `df.dropna(axis="index")`
6. `df.dropna(axis=1)` or `df.dropna(axis="columns")`
7. `SimpleImputer()`
8. `IterativeImputer()` and `KNNImputer()`
9. Probably `SimpleImputer(strategy="constant", fill_value="missing")` because its creates a new category for missings.
10. Probably `SimpleImputer` with `strategy="mean"` or `"median"` but with `add_indicator=True` because its creates a new column indicating the missing.

## 🔎 Outliers

1. It is a value that is a rare case. Observations that are far from the others.
2. Plotting the distribution showing the points, something like a boxplot or a stripplot. Then remove the outlier by clipping the variable.
3. Data is not polluted by outliers and we are interested in detecting whether a **new** observation is an outlier.
4. `svm.OneClassSVM`, `ensemble.IsolationForest`, `neighbors.LocalOutlierFactor` and `covariance.EllipticEnvelope`.


## 🖋 Typos

1. A typographical error.
2. The fuzzywuzzy package.



# Practical case

- "Street Number Suffix": Missing because it does not exist.
- "Zipcode": Missing because it was not recorded.

