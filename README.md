Data-preprocessing-python
=========================

This repository contains tools and tricks for processing data before analysis using ML algorithms. I follow Chapter 4 of Sebastian Raschka's Python Machine Learning.

How to handle missing data?
---------------------------

We can either eliminate the rows and columns with missing data, if there are a very high percentage of missing data. Alternatively, we can impute the values of missing slots using a mean, median, or most frequent value strategy. One needs to import the Imputer class from the sklearn.preprocessing module.  
We can also use Python libraries such as Numpy and Pandas to handle missing values.

Class labeling
--------------

Features can take nominal or ordinal values. One can encode class labels either manually or using the LabelEncoder from the sklearn.preprocessing module. Furthermore, there is one-hot encoding reserved for nominal features. This is done via OneHotEncoder class from the same sklearn.preprocessing module.

Splitting dataset into test and train subsets
---------------------------------------------

Datasets can be split into training and test sets via train_test_split function from the sklearn.cross_validation module.  

Normalizing/Standardizing features
----------------------------------

Features can be normalized or standardized using the MinMaxScaler or StandardScaler function from the sklearn.preprocessing module.

How to choose features of interest from the dataset?
----------------------------------------------------

L1 regularization can be used to yield sparse weight vectors in Logistic Regression (for example). This gives us a sense of the relative importance of features in classification. The algorithm which has a built in One-versus-Rest generalazition classifies between three types of wines as shown below. Note however, that the features of importance for each classification, i.e. wine 1 vs rest, wine 2 vs rest, wine 3 vs rest are different. Hence, universally important features might not exist!  

![](https://github.com/FyzHsn/Data-preprocessing-python/blob/master/L1RegularizationWine1.png?raw=true)  
![](https://github.com/FyzHsn/Data-preprocessing-python/blob/master/L1RegularizationWine2.png?raw=true)  
![](https://github.com/FyzHsn/Data-preprocessing-python/blob/master/L1RegularizationWine3.png?raw=true)  

**Feature selection** can be carried out using the Sequential Backwards Selection (SBS) algorithm. This is particulaly useful for models that do not allow regularization. On the other hand, **feature importance** can be carried out using the RandomForestClassifier algorithm. They are all included in the scripts. I have used the scripts by Raschka for practise. He himself has his book on his github account. 


