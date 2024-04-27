# Bayesian Linear Regression

Bayesian Linear Regression is an approach to linear regression where statistical analysis is applied within the context of Bayesian statistics. Instead of just fit a linear regression to the data, it goes a step further in treating parameters (intercept and slope) as random variables with their own distributions.

![Bayesian Linear Regression](https://miro.medium.com/max/1000/1*ILB9lKjCzRb_5Xfq_PwjXQ.png)

Linear regression is a common method to model the relationship between a dependent variable and one or more independent variables. Bayesian Linear Regression uses the Bayesian method to model and understand the relationship between the dependent and independent variables.

## How does Bayesian Linear Regression work?

The Bayesian Linear Regression model works by calculating the posterior distribution of the parameters. In the Bayesian approach, we start with a prior distribution for parameters 𝑎 and 𝑏, and the likelihood function which we get from the observed data is used to update our prior beliefs, to give the posterior distribution of 𝑎 and 𝑏.

```python
  # Implementing Bayesian Regression
  from sklearn import linear_model

  # Initialize Bayesian Ridge Regression
  bayes = linear_model.BayesianRidge()

  # Fit the Model to the training data
  bayes.fit(X_train, y_train)

  # Predict the response variable
  y_pred = bayes.predict(X_test)
```

## Advantages of Bayesian Linear Regression

1. **Responds to Uncertainty:** Bayesian Linear Regression has the advantage that it accounts for uncertainty in a way that general linear regression models do not.
2. **Flexibility:** The model is flexible to the prior distribution.
3. **Updates Easily with More Data:** Bayesian models can update the estimated probability easily with new data.

## Disadvantages of Bayesian Linear Regression

1. **Priors can be Biased:** The major disadvantage is heavy dependence on the choice of the prior. If the prior is wrongly chosen, then it takes much data to override that and come up to the true value.
2. **Computationally Expensive:** Calculating the posterior distribution involves multiple integral calculations which can be computational expensive.

## When to Use Bayesian Linear Regression

Bayesian Linear Regression is particularly useful when we have limited data or we expect certain correlation or patterns to hold. It is also ideal if your data involves complex relationships that you have a strong reason to believe follow a particular pattern.

The Bayesian approach becomes especially useful when we have the small dataset or multi-collinear features where linear regression can become unstable.

Sources: <br>

1. [Bayesian Linear Regression](https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7)
2. [Bayesian Ridge Regression](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression)
