# Passive Aggressive Classifier

A Passive-Aggressive Classifier is an online-learning algorithm that remains passive for a correct classification outcome, and turns aggressive in the case of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.

The Passive-Aggressive algorithms are called so because:

- Passive: If the prediction is correct, keep the model and do not make any changes. i.e., the data in the example is not enough to cause any changes in the model.
- Aggressive: If the prediction is incorrect, make some changes to the model. i.e., some change to the model may correct it.

![Passive Aggressive Classifier](https://miro.medium.com/max/546/1*7lugLKDS8vLS9muazao8Fw.png)

## Python Implementation

Here is a simple example:

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

data = fetch_20newsgroups()
X = data.data
y = data.target

vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)

clf = PassiveAggressiveClassifier()
clf.fit(X_vector, y)

predictions = clf.predict(X_vector)
print("Accuracy:", accuracy_score(y, predictions))
```

## Advantages

1. **Efficient**: Passive Aggressive Classifier is an online learning algorithm and therefore is very efficient on large datasets.

2. **Flexible**: It does not require a learning rate.

3. **Large Margin**: It naturally supports the max-margin principle which can help generalize the model.

## Disadvantages

1. **Non-probabilistic**: Passive Aggressive Classifier does not output probabilities.

2. **No Learning Rate**: Passive Aggressive Classifier does not include a learning rate. This means that the updates it makes can be quite drastic in the event of a misclassification, which can lead to overfitting if too much importance is given to these outliers.

## When to use Passive Aggressive Classifier

IBM’s Watson Natural Language Classifier, text classification, and spam filtering in Emails are some of the major applications of Passive-Aggressive Classifiers. It can also be implemented in cases where early prediction is essential, as in the case of live risk analysis in banks, real-time predictions of election results, etc. It's primarily used when data is arriving sequentially or the model needs to be changed dynamically with each new prediction, and you want to update your model as quickly as possible as computing resources are limited.
