
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
1) Data Processing/Cleaning/ Exploratory Data Analysis (EDA)
2) Machine Learning/Predictive Modeling
3) Results & Improvement
4) Writeup & Reporting

## Project Description

This project is a multiclassification problem, that can be used to help someone through tweets from Twitter, detect mental health issues. Each tweet is identified as 1 of 4 different mental health labels normal, anxious, stressed and lonely. All data for this project is pulled from the Kaggle competition "Tweet Mental Health Classification". In the training data set, only two features were used for this project the tweets (independent feature) and their mental health label (dependent feature). Once a predictive model is generated Kaggle has also provided a test data set, containing just the tweets, for the model to apply labels to. 

### Data Processing / Cleaning / EDA

The data cleaning for this project was straight forward. There was no missing data for this project and the tweets were preformatted as object type containing a string of words and characters. After getting an overview of the data set I removed any duplicate rows. 

The next stage was cleaning each tweet individually leaving critical words to build a predictive model off of.  The first step was removing any characters that were not in the English alphabet (a-z, A-Z). To make it easier to process each word and character I made each character lower case for uniformity. From here cleaning each word was done in two parts. The first removing what is referred to as “stop words”. These words are often transition or filler words between the critical words we would like to analyze, some examples are “and”, “it”, “the”, “a”, “an”, “in”, etc.) Once the stops words are removed each remaining word is reduced to its stem word. This will reduce the number of variations of the same word by removing the prefix, affix and tenses from the word. This will also help with Twitter language where words are often written short hand. Some examples of reducing to a stem word are owned to own, seizing to size, itemization to item, etc. 

Once the stop words are removed and each word is in its stem form, the output is a single list with each index representing each tweet row from the data set. With each row cleaned, used counter to see what words appeared more frequently for each label. To visually display the most common words for each label I used wordcloud. Looking at the top 20 most common words, negative connotative words appeared more frequently with the mental health labels stressed, anxious and lonely. The mental health label normal had more positive words then the other labels and more positive words than negative words 

### Machine Learning/Predictive Modeling

The predictive modeling process is split in to two parts text vectorization and using that data to evaluate each machine learning model (using hyperparameter tuning) to find the most accurate model to use with the test data. With each tweet cleaned leaving only the critical words, three different text vectorization methods were used to create a bag of words model (Count Vectorizer, Tfidf Vectorizer and Hashing Vectorizer). This turns my list of critical words in to a data set, where each word is a feature and every tweet is represented by the number of times that word appears in that row. For example the tweet “next time meet someon new dont ask ask love”, would have 8 feature columns (one for each different word). The row would be represented with the number 1 for all words except ask would have the number 2. This process is repeated for each tweet as the number of feature column grows the row would have 0s representing the words that do not appear in the row. For this project I used the top 5000 features. The process is used with fit transform to create the new independent features in the training data to use with the machine learning models. The new data set is referred to as a bag of words model. This full process is performed using natural language processing.

The new training data is divided in to train test split with the test size set to 30%. Four different machine learning (ML) models are then trained and evaluated with the data, Multinomial NM Algorithm, Passive Aggressive classifier, Linear support Vector Machine and Logistic Regression. Visualize results with confusion matrix to see true classification and false classification.


### Results & Improvement

First set of results using top 5000 words, vectorizer with fit_transform and tfidf transformation for count vectorizer -> accuracy 0.62613

In the data cleaning process clean out additional words that are not considered critical for this project’s multiclassification.
Additional hyperparameter tuning.
Additional methods for text vectorization or fit transformation.
Additional training models both machine learning and deep learning.

