#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Loading the Data
loans = pd.read_csv('loan_data.csv')

#Basic Exploration
loans.info()

#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 9578 entries, 0 to 9577
#Data columns (total 14 columns):
#credit.policy        9578 non-null int64
#purpose              9578 non-null object
#int.rate             9578 non-null float64
#installment          9578 non-null float64
#log.annual.inc       9578 non-null float64
#dti                  9578 non-null float64
#fico                 9578 non-null int64
#days.with.cr.line    9578 non-null float64
#revol.bal            9578 non-null int64
#revol.util           9578 non-null float64
#inq.last.6mths       9578 non-null int64
#delinq.2yrs          9578 non-null int64
#pub.rec              9578 non-null int64
#not.fully.paid       9578 non-null int64
#dtypes: float64(6), int64(7), object(1)
#memory usage: 1.0+ MB


loans.describe()

loans.head()

#Exploratory Data Analysis
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')

#The output graph for this line of code can be found at issue section of this repository - https://tinyurl.com/y9gf9jmd

#this time select by the not.fully.paid column.

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')

#The output graph for this line of code can be found at issue section of this repository - https://tinyurl.com/yd98cuuy

#we will show countplot using seaborn
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')

#The output graph for this line of code can be found at issue section of this repository https://tinyurl.com/y78lqahe


#Let's see the trend between FICO score and interest rate. 
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')

#The output graph for this line of code can be found at issue section of this repository - https://tinyurl.com/yar2a2d7


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
           
#The output graph for this line of code can be found at issue section of this repository - https://tinyurl.com/ybtee8k7


#Setting up the Data
#Let's get ready to set up our data for our Random Forest Classification Model!
#Check loans.info() again.

loans.info()


#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 9578 entries, 0 to 9577
#Data columns (total 14 columns):
#credit.policy        9578 non-null int64
#purpose              9578 non-null object
#int.rate             9578 non-null float64
#installment          9578 non-null float64
#log.annual.inc       9578 non-null float64
#dti                  9578 non-null float64
#fico                 9578 non-null int64
#days.with.cr.line    9578 non-null float64
#revol.bal            9578 non-null int64
#revol.util           9578 non-null float64
#inq.last.6mths       9578 non-null int64
#delinq.2yrs          9578 non-null int64
#pub.rec              9578 non-null int64
#not.fully.paid       9578 non-null int64
#dtypes: float64(6), int64(7), object(1)
#memory usage: 1.0+ MB


#Categorical Features

#Notice that the purpose column as categorical

#That means we need to transform them using dummy variables so sklearn will be able to understand them. 
#Let's do this in one clean step using pd.get_dummies.

cat_feats = ['purpose']

final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

final_data.info()

#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 9578 entries, 0 to 9577
#Data columns (total 19 columns):
#credit.policy                 9578 non-null int64
#int.rate                      9578 non-null float64
#installment                   9578 non-null float64
#log.annual.inc                9578 non-null float64
#dti                           9578 non-null float64
#fico                          9578 non-null int64
#days.with.cr.line             9578 non-null float64
#revol.bal                     9578 non-null int64
#revol.util                    9578 non-null float64
#inq.last.6mths                9578 non-null int64
#delinq.2yrs                   9578 non-null int64
#pub.rec                       9578 non-null int64
#not.fully.paid                9578 non-null int64
#purpose_credit_card           9578 non-null float64
#purpose_debt_consolidation    9578 non-null float64
#purpose_educational           9578 non-null float64
#purpose_home_improvement      9578 non-null float64
#purpose_major_purchase        9578 non-null float64
#purpose_small_business        9578 non-null float64
#dtypes: float64(12), int64(7)
#memory usage: 1.4 MB


#Train Test Split
#Now its time to split our data into a training set and a testing set!
#Use sklearn to split your data into a training set and a testing set 

from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


#Training a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

#Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.

dtree = DecisionTreeClassifier()

#Predictions and Evaluation of Decision Tree

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

#print(classification_report(y_test,predictions))
#
#             precision    recall  f1-score   support
#
#          0       0.85      0.82      0.84      2431
#          1       0.19      0.23      0.20       443
#
#avg / total       0.75      0.73      0.74      2874


print(confusion_matrix(y_test,predictions))
#[[1995  436]
# [ 343  100]]


#Training the Random Forest model
#Now its time to train our model!

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)


#Prediction and evaluation

predictions = rfc.predict(X_test)

#Now create a classification report from the results. 

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

#             precision    recall  f1-score   support
#
#          0       0.85      1.00      0.92      2431
#          1       0.57      0.03      0.05       443
#
#avg / total       0.81      0.85      0.78      2874


#Show the Confusion Matrix for the predictions.

print(confusion_matrix(y_test,predictions))
#[[2422    9]
#[ 431   12]]
 
 
