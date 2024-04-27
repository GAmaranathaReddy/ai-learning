# AdaBoost Classifier

AdaBoost (Adaptive Boosting) is a powerful ensemble machine learning algorithm that’s used to improve the performance of a model by combining several weak learners to produce a highly accurate prediction. First introduced by Freund and Schapire in 1996, the algorithm works sequentially by fitting the initial model to the data and then fitting subsequent models to the previously mis-classfied instances.

`AdaBoost` is best used to boost the performance of decision trees on binary classification problems.

```python
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(x_train,y_train)
```

![AdaBoost](https://miro.medium.com/max/1579/1*m2UHkzWWJ0kfQyL5tBFNsQ.png)

## How AdaBoost Works

1. The AdaBoost algorithm begins by first selecting a training subset randomly.

2. It then iteratively trains the AdaBoost machine learning model by selecting the training set based on the accurate prediction of the last training.

3. It assigns the higher weight to wrong classified observations so that in the next iteration these observations will get the high probability for classification.

4. Also, it assigns the weight to the trained classifier in each iteration according to the accuracy of the classifier. The more accurate classifier will get high weight.

5. This process iterate until the complete training data fits without any error or until reached to the specified maximum number of estimators (n_estimators).

6. To classify, perform a "vote" across all of the learning algorithms AdaBoost constructed.

## Advantages of AdaBoost

- AdaBoost is easy to implement. It iteratively corrects the mistakes of the weak classifier and improves accuracy by combining weak learners.

- AdaBoost is not prone to overfitting.

- No prior knowledge is needed about weak learners.

## Disadvantages of AdaBoost

- AdaBoost is sensitive to noisy data and outliers.

- It's performance depends on data and weak learner's selection.

- It requires enough Data.

- Computationally expensive.

## When to Use AdaBoost

The AdaBoost Algorithm is used for:

- Solving a variety of Complex problems in different fields such as biology, computer vision, and speech processing.

- Solving Binary and Multiclass classification problems.

- Conducting face detection as a binary classification; the parts of the image are either a face or background.

In general, AdaBoost is a powerful tool when you're working with large amounts of data to make predictions with binary outcomes. But do consider it might be more resource demanding and computationally expensive than simpler models.
