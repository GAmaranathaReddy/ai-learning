# Logistic Regression

Logistic Regression is a type of algorithm used in Machine Learning for binary classification problems. The outcome of this algorithm is discrete (not continuous). The results can be interpreted as probabilities of success.

Contrary to popular belief, the logistic regression model is a statistical model that uses a logistic function to model a binary dependent variable. In simpler words, it deals with situations where you need to predict an outcome that can have only two possible types of values.

The goal of logistic regression is to create a best-fit model to describe the relationship between the predictor and the response variable.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assume we have a 'df' DataFrame with 'target' as the prediction label.
x = df.drop('target', axis=1)
y = df.target

# Splitting our data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=123)

# Normalizing our data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Initialize our classifier
logistic_classifier = LogisticRegression()

# Fitting the model to train set
logistic_classifier.fit(x_train, y_train)

# Predicting the test set result
y_pred = logistic_classifier.predict(x_test)
```

The logistic regression function has a characteristic 'S' shape and it can seem complex at first, but it's rather simple when you understand it.

![Logistic Regression Graph](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

In the graph above, you can see the logistic function; it's an 'S' shaped curve that can take any real-valued number and map it into a value from 0 to 1. If the curve goes to positive infinity, y approaches 1, and if the curve goes to negative infinity, y will become 0.

If the output of logistic regression is higher than 0.5, we can conclude that the model predicts class 1. If logistic regression output is less than 0.5, then the model predicts class 0.

Though it is one of the simpler and older algorithms, Logistic Regression is still one of the most widely used algorithms due to its simplicity and the fact that it can be implemented quickly and provide very efficient predictions on binary classification problems.

## Advantages of Logistic Regression:

1. **Efficiency:** Computationally inexpensive compared to complex methods.

2. **Easily understandable output:** The probabilities provided can be interpreted as a risk, which is directly readable unlike other classifiers like SVMs or Random Forests whose outputs are difficult to interpret in probabilistic terms.

3. **Less vulnerable to overfitting:** Penalized logistic regression models can avoid overfitting by using built-in feature selection properties of the regularization technique itself.

4. **Works well with smaller dataset:** It performs well even if you have fewer training samples available for your model creation.

5. **Robustness:**
   - No assumption about distribution.
   - Robust against statistically irrelevant features (e.g., transformations).

## Disadvantages of Logistic Regression:

1. **Binary Targets only Suitable:**

   - Cannot predict continuous outcomes.
   - Outcomes must not have multi-class categories.

2. **Requires large sample size:** To achieve stable results, it requires at least 50 observations per predictor variable because maximum likelihood parameters are quite unstable unless sufficient details are present.

3. Controversy around interpretation when used with non-linear effects and interactions

   - Nonlinearity needs transformation which makes it difficult maintaining its original ease-of-use appeal.

4. Retains all outliers and influential values without adjustment

5. Vulnerable towards 'Perfect Seperation'

6. Provides less insight into individual predictors than decision trees

   Instructions regarding assumptions:

   - Logit should be linear in X(variable)
   - Independent errors(multicollinearity)

## Appropriate Usage Scenarios:

1.  When your target variable has binary outputs i.e., "yes" or "no", "true" or "false", etc., you may use Logistic JavaScript SDKs gression Classifier predictive analytics technique
2.  Use case scenarios include Email Spam/ Not-spam classifier; Churn Prediction,"Will customer buy this product?" , Health diagnosis such as Diabetes prediction: Will patient get diabetes?
3.  In credit risk modelling result will either be default or non-default class making it classic scenario for applying logistic regression classifier.

4.  Logistic regression is best suited for situations where data is cleanly separable linearly.

5.  The main advantage lies in its simplicity since it creates straightforward decision boundaries which align along axes within input space(This gives it property-term“linear classification”).

6.  Can handle several categorical variables-
    If you may want include categorical variables(example: gender,race)there’s no requirement convert these vars into numbers-their presence adequate enough deal with multiple categories deliver robust insights.
