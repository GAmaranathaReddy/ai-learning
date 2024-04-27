# XGBoost Regression

XGBoost stands for "Extreme Gradient Boosting" and it is an implementation of gradient boosting machines. The XGBoost is a popular machine learning algorithm used for regression and classification problems. This algorithm improve speed and model performance.

The XGBoost has an immensely high predictive power which makes it the best choice for accuracy in events as it reduces the error by following the gradient descent (like other boosting algorithms).

In XGBoost Regression, the output is the prediction which is continuous, like predicting a house price or the temperature, etc.

![XGBoost](https://www.researchgate.net/profile/Li-Mingtao-2/publication/335483097/figure/fig3/AS:934217085100032@1599746118459/A-general-architecture-of-XGBoost.ppm)

## The process of XGBoost Regression

XGBoost is an ensemble learning method. In ensembling, weak learners (models) are combined to form a stronger prediction rule. XGBoost used trees as weak learners.

For each tree, the algorithm identifies the best split point and variables, and leaves are assigned a real score. New functions are added to the existing functions until no further improvements can be made. Regularization parameters are also used in XGBoost to decrease overfitting and improve overall model.

## Advantages of XGBoost Regression

1. **Performance**: XGBoost dominates structured or tabular datasets on classification and regression predictive modeling problems.
2. **Speed**: XGBoost is really fast when compared to other implementations of gradient boosting.
3. **Scalability**: XGBoost scales on multi core CPU, it supports distributed computing which makes it feasible for large datasets as well.

## Disadvantages of XGBoost Regression

1. **Complexity**: XGBoost involves many parameters to tune and this makes it difficult for beginners to use it efficiently.
2. **Time consuming for larger datasets**: It takes a lot longer time to train the model compared to decision trees.
3. **Noisy data**: If the data is noisy(which doesn’t show a certain pattern), XGBoost will overfit for sure.
4. **Feature interpretation**: Since it uses gradient boosting algorithm which is an ensemble of decision trees, results of the model are not easily interpretable.

## Application scenarios for XGBoost Regression

1. **Predicting House prices**: Here the outcome to be predicted is the price of the house which is a continuous value. Therefore, Regression algorithm is used.
2. **Stock price prediction**: The stock price for the next day is a numeric quantity and is a target variable which can be predicted using the regression algorithms.

Here's how you can perform XGBoost Regression in Python:

```python
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load Dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize the XGBoost Regressor
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

# Fit the regressor with training data
xg_reg.fit(X_train, y_train)

# Predict
preds = xg_reg.predict(X_test)
```

And you can find the Root Mean Square Error this way:

```python
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
```
