# Poisson Regression

Poisson Regression is a form of regression analysis, which is utilized to model count data and contingency tables. You would use Poisson regression when the outcome you are interested in is counts. A Poisson regression model is sometimes known as a log-linear model, especially when used to model contingency tables.

Looking into the generalized linear models (GLMs), Poisson regression assumes the response variable Y has a Poisson distribution and assumes the logarithm of its expected value can be modeled by a linear combination of unknown parameters.

With that said, if:

` Y ~ Poisson (λ)`
`Log (λ) = η = Xβ`

A regression equation is a deterministic part and the stochastic (random) part. The deterministic part consists of the independent variable(s) and the parameter(s), including the intercept and the slope. The stochastic part shows the random variability around the deterministic prediction.

(lambda) λ stands for the Poisson distribution parameter or the expected count.

![Poisson Regression Formula](https://miro.medium.com/max/1050/0*FdDrGAs5VajWUeal.png)

## Python Implementation

Step 1: Import your needed libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
```

Step 2: Import your data. Let's use a dataset from seaborn for instance:

```python
# import necessary libraries
import seaborn as sns

#Load dataset
df = sns.load_dataset('tips')
df.head()
```

Step 3: Define X and y and fit the model using sm.GLM:

```python
X = df[['total_bill', 'size']]
y = df['tip']

# add constant to predictor variables
X = sm.add_constant(X)

# fit Poisson regression model
model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# view model summary
print(model.summary())
```

## Advantages of Poisson Regression

1. The main advantage of Poisson regression is its ability to handle count data. Count data often have unique features that are best captured by Poisson distribution.

2. It can establish relationships between a range of predictor variables.

3. It's an easy-to-understand model, even persons with a basic understanding of regression can interpret the result.

## Disadvantages of Poisson Regression

1. Since Poisson regression requires count data, its utility is limited in situations where outcome variables are not counts.

2. The model has strict assumptions, such as the mean and variance of the outcome variable are identical.

3. It can't handle negative numbers. You would need to use different regression modeling techniques in this case.

4. Poisson regression can be influenced by over-dispersion and under-dispersion, which might lead to incorrect inferences.

## When to Use Poisson Regression?

The assumption of Poisson regression is that the dependent variable follows a Poisson distribution. Therefore, you can use it in scenarios when your dependent variable is a count and a number of trials are known, and success probability is small.

Essentially, when modeling events that happen randomly but with a known average rate, Poisson Regression is the ideal choice. This includes transportation vehicle crashes, disease incidences, and customers calling a help-center.
