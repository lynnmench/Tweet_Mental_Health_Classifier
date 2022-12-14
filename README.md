
Author: Lynn Menchaca

Date: 09Dec2022


# Tweet_Mental_Health_Classifier

The purpose of this project is design a model to use tweets to classify mental health.


#### -- Project Status: Active

## Project Overview
### Resources
- Kaggle Data Set: [Tweet Mental Health Classification](https://www.kaggle.com/competitions/tweet-mental-health-classification)
- Youtube: Ken Jee -> Data Science Project from Scratch - [Part 4 (Exploratory Data Analysis)](https://www.youtube.com/watch?v=QWgg4w1SpJ8&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t&index=4)
- [plot confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
- [Multi-Class Text Classification Model Comparison and Selection](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568) by Susan Li

### IDEs/Language/
* Jupyter
* Python
* Pandas
* Sklearn
* Matplotlib
* Nltk

### Vectorizers
* Count Vectorizer with Tfidf Transformation
* Tfidf Vectorizer
* Hashing Vectorizer

### Machine Learning Models (using Hyperparameter Tuning)
* Multinomial NM Algorithm (with Tfidf Transformation)
* Passivie Aggressive Classifer
* Linear Support Vector Machine
* Logistic Regression


### Project Outline
1) Data Processing/Cleaning
2) Exploratory Data Analysis (EDA)
3) Machine Learning/Predictive Modeling
4) Results & Improment
5) Writeup & Reporting

## Project Description

Two data files are used for this project, a training file and test file. All data for this project is pulled from the Kaggle compitition "Tweet Mental Health Classification". 

Tweets are grouped by 4 mental health labels Normal, Anxious, Stressed, Lonely


Each tweet 


The task is to classify tweets into 4 different categories. The tweets were scraped using Tweety API. The data contains a train.csv which contains labelled data to train.

This is a straight forward classification task.


### Data Processing/Cleaning

Minimal cleaning was required for this project. 

