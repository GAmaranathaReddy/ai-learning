# Classification

Machine Learning classification models are a category of methods used to predict the category of a data point. These models are mainly used in the field of medical imaging, speech recognition, and many others. This page entails explanations with Python examples of popular Machine Learning Classification models.

1. [Logistic Regression](./LogisticRegression.md)
2. [K-Nearest Neighbours](./knn.md)
3. [Support Vector Machines](./SupportVectorMachines.md)
4. [Naive Bayes](./NaiveBayes.md)
5. [Decision Tree Classification](./DecisionTree.md)
6. [Random Forest Classification](./RandomForest.md)
7. [Stochastic Gradient Descent Classifier (SGD)](./sgd.md)
8. [Gaussian Process Classification (GPC)](./gpc.mg)
9. [Gradient Boosting Classifier](./gbc.md)
10. [AdaBoost Classifier](./adaboost.md)
11. [Bagging Classifier](./bagging.md)
12. [Extra Trees Classifier](./extra_tree.md)
13. [Passive Aggressive Classifier](./passive_aggre.md)
14. [Ridge Classifier](./ridge.md)

## Types of Classification Algorithms

1. **Discriminative Learning Algorithms**

   These algorithms try to learn the probability of an end result **Y** for a given feature set **X**. These algorithms try to determine how **Y** is directly a function of **X**. Mathematically these are shown as

   ![pyx](http://mathurl.com/cqd6fro.png)

   Some of these algorithms try to learn a hypothesis that tries to predict the possible classes, mathematically represented as

   ![binarhypothesis](http://mathurl.com/ya46sgrc.png)

2. **Generative Learning Algorithms**

   This type of Algorithms try to learn, the probability of a given set of features **X** for a particular class **Y** (mathematically represented as ![pxy](http://mathurl.com/bwse6yv.png)) and also, the probability of occurrence of this class **Y** (the probability of occurrence of a given class is represented as ![py](http://mathurl.com/byg852g.png) and is called **class prior**. The most popular example of such algorithms is the [Naive Bayes Algorithm](./04-NaiveBayes.md).
