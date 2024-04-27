# Naive Bayes Classifier

The Naive Bayes Classifier is a type of probabilistic machine learning model used for large scale classification problems. It assumes strong independence between features. It is simple but powerful algorithm for predictive modeling.

![Naive Bayes](https://miro.medium.com/max/875/1*39U1Ln3tSdFqsfQy6ndxOA.png)

## Basic Principle of Naive Bayes Classifier

Naive Bayes classifier apply Bayes' theorem with strong (naive) independence assumptions between the features. Let’s understand the Bayes Theorem.

### Bayes' Theorem

Bayes’ Theorem finds the probability of an event occurring given the probability of another event that has already occurred. Bayes’ theorem is stated mathematically as the following equation:

```
P(A|B) = P(B|A) * P(A) / P(B)
```

where A and B are events.

Here,

- P(A|B) is Posterior probability: Probability of hypothesis A on the observed event B.

- P(B|A) is Likelihood probability: Probability of the evidence given that the probability of a hypothesis is true.

- P(A) is Prior Probability: Probability of hypothesis before observing the evidence.

- P(B) is Marginal Probability: Probability of Evidence.

## Implementing Naive Bayes with Python (Scikit-Learn)

Let's implement a simple demonstration of Naive Bayes classifier using Scikit-Learn library's `GaussianNB` module.

### Importing Required Libraries

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn import metrics
```

### Load Dataset

```python
wine = datasets.load_wine()
```

### Split the Data

```python
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)
```

### Generate a Model

```python
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)
```

### Evaluating the Model

```python
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
```

The output will give you the accuracy of the model. Generally, Naive Bayes classifier gives high accuracy and is very fast in prediction.

## Advantages

1. **Ease of implementation:** With the use of strong assumptions, implementation becomes straightforward and fast.

2. **Efficiency:** They require a small number of training data to estimate the parameters necessary for classification because they are capable using maximum likelihood estimates.

3. **Fast and Scalable:** Naive Bayes classifiers are extremely fast compared to more sophisticated methods.

4. **Good Performance in Multi Class Prediction** When it comes onto handling multiple classes, naive bayesian models give better results as compared to others like logistic regression etc.

5. **Perform well with categorical input variables** This makes it highly optimal for text categorization problems which involves categorical variable inputs.

## Disadvantages

1.  **Conditional Independence Assumption**: One main limitation of Naive Bayes is the assumption of conditional independence between all pairs of predictors.

## Appropriate Usage Scenarios

- **Document classification and spam filtering** :Naïve Bayesian model performs particularly well in case of document classification and spam filtering.

- **Real time prediction** : Since this algorithm helps predict multiclasses, it can also be used where we need probablity outputs like weather predictions or predicting diseases etc.

- **Text Classification** : Its high speed and ability handle huge dataset effectively make it perfect choice Surprisingly despite its over-simplified assumptions,

- **Recommendation System**: Though recommendation system uses complex techniques but at heart some form if naive base is definitely present for initial recommendations where you don't have enough data about likes/dislikes/preferences about users both from his/her behaviour or product side/filters/content descriptions/details/collaborative details/other decision making parameters or classifiers

## Conclusion

The Naive Bayes classifier is a simple and effective classification algorithm, being particularly useful when working with text data and high dimensional datasets. It relies on the Bayesian theorem and assumes all variables to be independent in order to predict the class.

In practice, the independence assumption is often violated, but Naive Bayes classifiers still tend to perform very well under this unrealistic assumption. Especially for small sample sizes, Naive Bayes classifiers can outperform the more powerful alternatives.
