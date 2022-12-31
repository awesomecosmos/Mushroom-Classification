# Mushroom Classification
 
This is a fun project to apply the Exploratory Data Analysis (EDA) process and numerous classification algorithms on the [Mushrooms dataset, available from Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification?datasetId=478&sortBy=voteCount), which consists of 8143 data observations of mushrooms and 23 features that describe two classes of mushrooms - edible, and poisonous. 

<img src="img/mushrooms.jpg">

# Key finding: The K-Nearest Neighbors model (accuracy 99.63%) or the linear Support Vector Classifier model (accuracy 97.27%) should be used to classify poisonous mushrooms! Also pay attention to a mushroom's veil color, gill size, bruises, and ring type, since they can be indicators of a poisonous mushroom!

## Authors

- [@awesomecosmos](https://www.github.com/awesomecosmos) (me!)

## Table of Contents

  - [Introduction](#introduction)
  - [Data Source](#data-source)
  - [Analysis Methods](#analysis-methods)
  - [Algorithms](#algorithms)
  - [tl;dr - Results](#tldr-results)
  - [Discussion](#discussion)
  
## Introduction

I love going out for walks in nature. Often in the parks and forested areas I wander during these walks, I come across mushrooms growing here and there (which are fun to squish sometimes tbh). I have noticed these mushrooms come in a variety of shapes and sizes, and that made me wonder about them. Natuarally as a data scientist, I headed online to see if I could find a dataset on mushrooms that I could analyze. This project is the resulting breakdown of analyzing what (in general) makes mushrooms poisonous vs edible. Read on for more info!

## Data Source

I found a [mushrooms dataset on Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification?datasetId=478&sortBy=voteCount), which consists of 8143 records of mushrooms, broken down by 23 characteristics that describe the mushrooms, and their classification (either 'e'=edible, or 'p'=poisonous). This turned out to be a remarkably clean dataset, interestingly enough! Here is some more information about the dataset, taken directly from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification?datasetId=478&sortBy=voteCount):

> This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.

> Time period: Donated to UCI ML 27 April 1987

## Analysis Methods

Given that the mushrooms in the dataset are either classified as 'edible' or 'poisonous', this indicates this is a binary classification problem. Therefore, I have structured my analysis in two main parts:

1. Exploratory Data Analysis
2. Predicting classifications of mushrooms

I have performed my analysis in a Python Jupyter notebook, [mushroom_classification.ipynb](https://github.com/awesomecosmos/Mushroom-Classification/blob/main/mushroom_classification.ipynb).

## Algorithms

In order to predict the binary classifications, I used a range of algorithms, and chose the best algorithm (discussed later). Here are the algorithms I used:
- Gaussian Naive Bayes
- Random Forest
- Decision Tree
- Logistic Regression
- SVC (Support Vector Classification)
- K-Nearest Neighbors
- XGBoost

## TLDR Results

Of the seven algorithms that were used to classify the mushrooms, 3 had overfitted the model, and so were discarded for final analysis. Of the four models that remained (Naive Bayes, Logistic Regression, SVC, and K-Nearest Neighbors), KNN and SVC were found to have the highest precision and recall (>96%). The most important features that contribute to a mushroom being poisonous according to the SVC model were found to be the veil color, gill size, bruises, and ring type of the mushrooms.

## Discussion

For this project, our aim is to find the best possible algorithm which classifies a mushroom as either edible or poisonous. We used 7 algorithms and evailuated their performances. Now we will discuss a few evaluation metrics to judge the best algorithm that should be used to predict mushroom classifications. But first, some terminology! Most evaluation metrics are defined in terms of positives and negatives, as seen in the confusion matrices. In our confusion matrix for our binary classification problem, a positive is defined as 1, which corresponds to poisonous mushrooms. Therefore, the negative class corresponds to 0, which are edible mushrooms. This tells us that our models are finding which mushrooms are poisonous (which is the hypothesis), rather than the other way round. Therefore, here is some more terminology before we define the evaluation metrics:

- TP (True Positives): how many data points were correctly classified as poisonous (actual = 'p', predicted = 'p').
- FP (False Positives): how many data points were incorrectly classified as poisonous (actual = 'e', predicted = 'p').
- FN (False Negatives): how many data points were incorrectly classified as edible (actual = 'p', predicted = 'e').
- TN (True Negatives): how many data points were correctly classified as edible (actual = 'e', predicted = 'e').

**Accuracy**
Accuracy is defined as follows: $\frac{TP+TN}{TP+TN+FP+FN}$

- The accuracy metric is good for a balanced dataset (which we have), and for when every class is important.

**Precision**
Precision is defined as follows: $\frac{TP}{TP+FP}$

- The precision metric is good for measuring how often class 'p' is indeed classified as class 'p', i.e. maximizing on TPs. 
- In our case, this is a good metric to compare our models on, since we need to correctly predict poisonous mushrooms as poisonous, instead of edible, since it can lead to bad things for humans!

**Recall**
Recall is defined as follows: $\frac{TP}{TP+FN}$

- The recall metric is good for measuring how often class 'e' is indeed classified as class 'e', i.e. maximizing TNs.
- This is also a good metric to compare our models, since we also want correctly-classified edible mushrooms!

**F1 Score**
F1 Score is defined as follows: $2\times \frac{precision\times recall}{precision + recall}$

- The F1 metric is referred to as a 'harmonic mean between precision and recall'. 
- This means that it is a good metric to measure how often edible mushrooms and poisonous mushrooms are each correctly classified.

**ROC/AUC Score**
- The ROC (Receiver Operating Characteristic) curve is a graph of the TP rate vs the FP rate.
- The AUC (Area Under Curve) score is a measure of the area under the ROC curve.
- The ROC/AUC score is good for measuring the probability of good predictions made for both edible and poisonous mushrooms.


After looking at all these metrics, we will define our primary metric as precision, since we are interested in minimizing the chance of eating poisonous mushrooms, which will only happen if most of the poisonous samples in the dataset are correctly classified.

One more thing before we start evaluating our models - we note that a few algorithms provided 100% on each metric. This shows that these models have likely overfitted, i.e. the model learned specific rules from the training data and was able to apply those to the testing dataset as-is. We will disregard these overfitted models (Random Forest, Decision Tree, K-Nearest Neighbours, and XGBoost) for final evaluation.

Ok, now we have enough background to start evaluating our models. Once more, let's look at the performance metrics table, after filtering out the overfitted models, based on accuracy.

Based on our primary metric, precision, we find that the KNN model has performed the best, followed by SVC (linear kernel), followed by logistic regression. The least precise model was Naive Bayes. The KNN model provided 99% precision, SVC provided 98% precision, logistic regression provided 94% precision, and finally Naive Bayes provided 90% precision. This indicates that the KNN model classified 99% of all poisonous mushrooms as poisonous.

We find that the other metrics also have similar performances, with KNN performing the best, followed by SVC, then logistic regression, then Naive Bayes. However, in terms of recall, we find that KNN has 100% recall, which shows that it correctly classified 100% of edible mushrooms as edible. Interesting!

**Based on the extensive analysis performed and comparison of metrics, we will conclude that a K-Nearest Neighbours algorithm, or a Support Vector Classifier model should be used to classify this dataset.**

So then what are the features that make a mushrooms poisonous anyway? According to the SVC model, the **top features that contribute to a mushroom being poisonous are its veil color, gill size, bruises, and the ring type**. The features that contributed least to the classification include gill spacing, stalk root, stalk shape, gill attachment, and ring number, just to name a few.

Therefore, to conclude this analysis, if I had an algorithm with me while I was in the forest exploring mushrooms that told me the likelihood of a mushroom being poisonous or not, I would go with either the KNN algorithm, or the SVC algorithm. I would also pay close attention to the veil color, gill size, bruises, and ring type of the mushrooms to corroborate the algorithm's predictions. ¯/\_(ツ)_/¯
