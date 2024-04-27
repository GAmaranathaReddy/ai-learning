# Random Forest Classifier in Python

Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean prediction of the individual trees. Random Forests corrects for decision trees' habit of overfitting to their training set.

Random forests are a powerful tool with several advantages:

- They are very easy to use because they require very few parameters to set and they are easy to understand.
- An efficient method for estimating missing data and maintains accuracy when a large proportion of the data are missing.
- It has an effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing.

![Random Forest Classifier](https://miro.medium.com/max/1400/1*58f1CZ8M4il0OZYg2oRN4w.png)

## Python Implementation

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Create a random forest Classifier
clf=RandomForestClassifier(n_estimators=100)

# Train the model using the training sets
clf.fit(X_train,y_train)

# Predict the response for test dataset
y_pred=clf.predict(X_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
```

In this example, we are importing the RandomForestClassifier from sklearn.ensemble. After importing we are creating an object of RandomForestClassifier class. The RandomForestClassifier takes several parameters such as n_estimators, max_features, etc. After creating an object of RandomForest Classifier we are fitting our model on train data. The fit method takes two parameters as arguments namely, X_train and y_train. After training our model, we are predicting the target values for the test data. By using metrics.accuracy_score method, we are finding accuracy score which is around 98%.

**Remember, one thing a Random Forest algorithm is just an extension of decision tree and it is just a bunch of decision trees. More the number of decision trees in a single predictor, better the prediction will be.**

However, keep in mind that increasing number of trees in the predictor is a bit computationally expensive.

## Advantages of RandomForest Classifier:

1. **Ease of Use:** The Random forest algorithm can handle missing values, so there's no need for imputation.
2. **High Accuracy:** Random forests generate higher accuracy than decision trees as it uses several trees rather then relying on individual decision trees.
3. **Data Versatility:** It can be used for both regression and classification type problems.
4. **Prevents Overfitting:** By aggregating the results from different trees, this reduces overfitting problem in decision trees hence delivering more accurate results.

## Disadvantages of RandomForest Classifier:

1. **Computationally Expensive**: This algorithm require much computational resources, as it builds numerous complexly structured models (trees).
2. **Time Consuming**: While executing on large datasets, training requires high computational power and may be time consuming.
3. **Complex Model**: Interpretability could be an issue because while random forect provides great predictive power but has very less interpretability.

## Appropriate Usage Scenarios:

- Predicting if an email is a spam or not.
- Predicting whether a given patient record (of features like age, bmi etc.) will get diabetes or not
- In Bioinformatics it is widely used for genome-based disease prediction .

In conclusion `RandomForestClassifier`, just like any other classifier has its pros and cons yet remains one of most effective classifiers with high performance across many tasks & domains!
