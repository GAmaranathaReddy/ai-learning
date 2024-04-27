# Negative Binomial Regression

Negative binomial regression is a type of generalized linear model (GLM), where the target variable is a count of the number of times an event occurs. The fundamental condition is the target variable or the response variable should be a count, such as the number of emails you receive each day, the number of customer complaints, etc.

When the variance is larger than the mean, data scientists deploy negative binomial regression. The mathematical form of the negative binomial distribution usually contains a dispersion parameter that adjusts the variance independently from the mean.

## Fundamental Concept

Negative binomial regression is used when the dataset, containing the number of occurrences of an event, shows variances that grow with the mean as well as the presence of over-dispersion.

Here is the formula:

![img](https://www.statsmodels.org/stable/_images/math/74540154548c2f67095be2ad6a40b2d65f06e7e0.png)

_Here, mu is the expected value of the response, and alpha is the dispersion parameter._

![img](https://miro.medium.com/max/641/1*-pZO1N_FmAoMKuZKDp9lEg.png)

## Advantages

- Negative binomial regression retains the efficiency and flexibility of ordinary regression.
- It allows the modeling of non-integer-based events and adjusts the variance independently from the mean.
- It is especially helpful for over-dispersed data, where the variance is greater than the mean.

## Disadvantages

- One disadvantage of negative binomial regression is that it can easily misspecified or misinterpreted.
- It has high sensitivity against the outliers.
- It requires a large sample size to achieve sufficient power.

## Code Snippet in Python

Here is a simple example of a negative binomial regression in Python.

```python
import statsmodels.api as sm
import numpy as np

data = sm.datasets.get_rdataset("mtcars").data
data["constant"] = 1
exog_vars = ["constant","mpg","hp","qsec"]
exog = sm.add_constant(data[exog_vars])
endog = data["am"]

mod = sm.GLM(endog, exog, family=sm.families.NegativeBinomial())
res = mod.fit()
print(res.summary())
```

In this example, dataset "Mtcars" is used. EXOG refers to the predictor variables (mpg, hp, qsec). ENDOG is the dependent variable ("am").

## Scenario for usage

Generally, negative binomial regression is used in scenarios where the data consists of count of events and the variance is greater than the mean. For example, it could be used in predicting the number of times a website will be visited in a given time frame.

Moreover, negative binomial regression can be used in various fields like predicting number of traffic accidents, disease counts in epidemiology, and so on.
