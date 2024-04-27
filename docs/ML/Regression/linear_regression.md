# Linear Regression: A Detailed Explanation with Python Implementation

Linear regression is a statistical model that examines the linear relationship between two (Simple Linear Regression ) or more (Multiple Linear Regression) variables — a dependent variable and independent variable(s). Linear relationship basically means that when one (or more) independent variables increase (or decrease), the dependent variable increases (or decrease) too.

![Linear Regression](https://miro.medium.com/max/1280/1*4dv_1BofQzjzM_IQQo4B6Q.png)

Linear Regression establishes a relationship between dependent variable (Y) and one or more independent variables (X) using a best fit straight line (also known as **regression line**).

## Math behind Linear Regression

It is represented by an equation `Y=a+b*X + e`, where `a` is the intercept, `b` is the slope of the line, and `e` is the error term. The equation can be used to predict the value of the target variable based on given predictor variable(s).

The intercept `a` is the predicted value of Y when the X is 0. The slope `b` is the predicted increase in Y resulting from a one unit increase in X. The error `e` is the difference between the actual value and the predicted value of the target variable.

![Equation](https://miro.medium.com/max/1400/1*eeIvlwkMNG1wSmj3FR6M2g.gif)

## Python Code Example

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Create linear regression object
lr = LinearRegression()

# Split the dataset into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

# Train the model using the training sets
lr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = lr.predict(x_test)
```

## Advantages of Linear Regression

1. It is a fairly simple model that doesn’t require high computation power.
2. Linear regression analysis, when used with statistical techniques, can help to determine the reliability of the predictions.
3. It is useful when the dataset is linearly separable.

## Disadvantages of Linear Regression

1. It assumes a linear relationship between variables. If the relationship is non-linear, it might produce a high bias error.
2. It's can be sensitive to outliers.
3. It might overfit or underfit the data depending on the nature of the dataset.

## When should you use Linear Regression?

Linear regression should be used when the target variable is of a continuous nature (e.g. Sales, Price etc.) and there is a linear relationship between the dependent (target variable) and independent variables.

In business it could be used for sales forecasting, resource consumptions forecasting, supply cost forecasting, etc.

####
