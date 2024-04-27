# Logistic Regression Classifier in Python

Logistic Regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, and uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’.

![Logistic Regression](https://miro.medium.com/max/724/1*UgYbimgPXf6XXxMy2yqRLw.png)

In the below sections, we will discuss more about Logistic Regression, along with its Python implementation.

## Python Explanation

```python
# Import library
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Create logistic regression object
logReg = LogisticRegression()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the training sets
logReg.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = logReg.predict(X_test)
```

## Advantages

1. **Easy to Implement:** Logistic regression is easy to understand and less complex than other algorithms.
2. **Speed:** It is less intense computationally, thus making it faster.
3. **Probabilistic Approach:** Logistic Regression provides a probability score for observations.
4. **Useful for Feature Importance:** Logistic regression is not a black box, you can understand what variable contribute to the outcomes and by how much.

## Disadvantages

1. **Binary Outcome Only:** Logistic regression is trying to predict a binary outcome. For multi-class problems, one will need to tweak the logistic regression model or turn to other algorithms.
2. **Sensitive to Outliers:** Logistic regression can be sensitive to outliers and may produce a biased model.

## Scenarios to Use

1. **Medical Fields:** Logistic Regression is used in certain fields, such as medical fields, where outcomes are often binary (e.g., patient survived/did not survive).
2. **Credit Scoring:** Logistic Regression is also used in credit scoring when the outcome is whether a customer will default or not.
3. **Marketing Applications:** When predicting if a customer will buy a product or not.
4. **Anywhere Probability is Needed:** Logistic regression is useful anywhere we need probability as an outcome.

## Conclusion

Logistic Regression has a wide variety of applications and is a powerful tool when used correctly. Despite its simplicity, it can provide a good baseline in many applications. It is essential to understand the assumptions and limitations of any algorithm before usage and Logistic Regression is no exception.
