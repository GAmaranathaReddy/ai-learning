# Polynomial Regression

Polynomial Regression is a form of regression analysis in which the relationship between the independent variable `x` and the dependent variable `y` is modelled as an nth degree polynomial. Polynomial regression fits a nonlinear relationship between the value of `x` and the corresponding conditional mean of `y`.

![Polynomial Regression](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Polyreg_scheffe.svg/2560px-Polyreg_scheffe.svg.png)

## Python Explanation

Polynomial Regression in python can be implemented by importing the `PolynomialFeatures` class from sklearn.preprocessing library. This class is used to convert the original features into polynomial features. Then the polynomial features are fed into a linear regression classifier for prediction.

Here's a basic example of how it works:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create polynomial features
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Feed it into Linear Regression
model = LinearRegression()
model.fit(x_poly,y)

# Predict
predictions = model.predict(x_poly)
```

## Advantages of Polynomial Regression

1. Broad range of function can be fit under it.
2. Polynomial basically fits wide range of curvature.
3. Polynomial provides the best approximation of the relationship between the dependent and independent variable.

## Disadvantages of Polynomial Regression

1. These are too sensitive to the outliers.
2. The presence of one or two outliers in the data can seriously affect the results of the nonlinear analysis.
3. In addition there are unfortunately fewer model validation tools for the detection of outliers in nonlinear regression than there are for linear regression.

## When to use Polynomial Regression

Polynomial Regression can be used when the data points closely follow a sensed non-linear trend. The trend which suggests that the data points are more likely fitted in a non-linear line, in that case Polynomial Regression can be used.

In which scenarios:

- When we try to model the yield of a chemical synthesis in terms of temperature,
- Or steam usage rate in terms of the settings of different heating dials.

These scenarios help in finding the real non-linear relationship between temperature and yield, and sometimes these relationships are not linear but they are quadratic or cubic.

Remember, it’s your job to test different polynomial degrees and make sure you are avoiding overfitting. Proper tools such as cross-validation and validation curves are there to help you. Please be responsible when creating your models.
