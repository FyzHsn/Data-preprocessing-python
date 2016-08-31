# -*- coding: utf-8 -*-
"""
In this script, I follow chapter 4 of Sebastian Raschka\'s Python Machine 
Learning. This chapter deals with the preprocessing of data and getting it 
ready for the machine learning algorithms. The main topics are:
1. Dealing with missing data
2. Handling categorical data
3. Partitioning dataset into training and test subsets.
4. Bringing features on the same scale.
5. Selecting meaningful features.
6. Regularizing data.

"""
import numpy as np
import pandas as pd
from io import StringIO

# create CSV (comma-separated values) dataset
csv_data = '''A, B, C, D
1.0, 2.0, 3.0, 4.0
5.0, 6.0,, 8.0
10.0, 11.0, 12.0,
,,,'''

# convert to data frame
df = pd.read_csv(StringIO(csv_data))

# we can see which values are missing in the data frame. for larger dataset,
# we can use an isnull method.
df.isnull().sum()

# to see the values of the df data frame, use .values function
print(df.values)

"""
Eliminating samples or features with missing values.

"""
# eliminating rows with missing data
print(df.dropna())

# eliminating columns with missing data
print(df.dropna(axis=1))

# eliminating columns where all values as NaN
print(df.dropna(how='all'))

# eliminating rows that have atleast 4 non-NaN values
print(df.dropna(thresh=4))

# only drop rows where NaN appears in specific columns
# for some reason, this command is not working. Any column name other than A
# gives and error. WHY???
df.dropna(subset=['A'])

"""
Imputing missing values. strategy can be mean, median and most_frequent.

"""
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)

"""
Handling categorical data. It can be nominal features which don't imply any
order in the context or ordinal features that can be ordered and sorted.

"""
df = pd.DataFrame([
            ['green', 'M', 10.1, 'class1'],
            ['red', 'L', 13.5, 'class2'],
            ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']   
print(df)         

# Let us map the ordinal features via the relationship, XL = L + 1 = M + 2
size_mapping = {
                'XL': 3,
                'L': 2,
                'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

# in order to transform the integer values back to the original mapping
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
print(df)

"""
Encoding class label: Though many of the estimators convert class labels to 
integers internally, like with the iris dataset, 

"""
class_mapping = {label:idx for idx, label in 
                 enumerate(np.unique(df['classlabel']))}
print(class_mapping)                 

# Encoding class labels via map()
df['classlabel'] = df['classlabel'].map(class_mapping)
df['size'] = df['size'].map(size_mapping)
print(df)

inv_class_mapping = {v:k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

# LabelEncoder from the preprocessing module can do the equivalent job
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

# We can transform the integer class labels back
print(class_le.inverse_transform(y))

"""
Performing one-hot encoding on nominal features. Ordinal size features can be
converted into integers. For nominal values, however, attaching integer values
can lead to bad performance of the algorithm, since it thinks that integers are
ordered. The workaround is to use a technique called one-hot encoding.

"""
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)
 
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())
print(pd.get_dummies(df[['price', 'color', 'size']]))

"""
Splitting datasets into test and train groups. Having dealt with the Iris data-
set, we now turn our attention to the wine data set. We will preprocess the 
dataset followed by splitting it into test and train subsets. 

"""
# download data from repository
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-\
databases/wine/wine.data', header=None)

# name columns
df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alkanalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavonoids',
                   'Nonflavonoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines',
                   'Proline']
                   
# look at all possible class labels - 3 possibilities                   
print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())

# check for na values - no missing values
print(df_wine.isnull().sum())

# Split wine data into test and train sets
from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0)
               
# standardize dataset using the MinMaxScaler        
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)     

# standardize dataset using the StandardScaler
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler() 
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test) 

"""
L1 vs L2 regularization: So far we have seen the usage of L2 regularization in
the joint minimization of cost function and weight magnitude penalty. This works
well to regularize the data and prevent overfitting by minimzing the weight
magnitude. Another option is the L1 regularization, which leads to sparse 
weight vectors, i.e. most entries are 0. This can be extremely useful for 
feature selection. Next up, is some code for Raschka that shows these points.

"""
from sklearn.linear_model import LogisticRegression

# logistic regression using L1 regularization
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy: ', lr.score(X_train_std, y_train))
print('Test accuracy: ', lr.score(X_test_std, y_test))

# Since the test and training scores are close, there is no indication of 
# overfitting. Find the weight coefficients. Remember One-versus-Rest is used.
print(lr.coef_)

"""
Plotting the weight components for each feature as a function of regularizat-
ion strength. When a weight vector is 0, it tells us that the feature is not
important. There is a caveat, however. If two features are correlated, and 
yet important, one of the weights could end up being zero. So, need to watch
out for that.

"""
import matplotlib.pyplot as plt

# create a figure object
fig = plt.figure()

# create an axis object in the figure: Fig 1 in a 1 by 1 grid
ax = plt.subplot(111) 

# colors corresponding to each feature
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

# initialize array to capture weight coefficients for varying C
weights, params = [], []

# consider C = 10**-4 to 10**5. c = -4..5
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', 
                            C=10**c, 
                            random_state=0)
    lr.fit(X_train_std, y_train)
    
    # interested in weights corresponding to wine 1 vs wines 2 and 3
    weights.append(lr.coef_[0])                            
    params.append(10**c)

# turn weight and parameter arrays into numpy arrays
weights = np.array(weights)
params = np.array(params)    

print(weights.shape[1])
    
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column+1],
             color=color)
             
plt.ylabel('Weight coefficients')
plt.xlabel('C')
plt.title('Wine 1 vs rest')
plt.legend(loc='upper left')             
plt.axhline(0, color='black', linestyle='--', linewidth=3)
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.xscale('log')
plt.savefig('L1RegularizationWine1.png')
plt.clf()             






