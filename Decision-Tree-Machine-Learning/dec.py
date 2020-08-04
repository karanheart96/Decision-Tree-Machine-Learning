#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Load the Data
df = pd.read_csv('kyphosis.csv')
df.head()

# 	Kyphosis 	Age 	Number 	Start
#0 	absent 	71 	3 	5
#1 	absent 	158 	3 	14
#2 	present 	128 	4 	5
#3 	absent 	2 	5 	1
#4 	absent 	1 	4 	15

#Now we will perform exploratory data analysis
sns.pairplot(df,hue='Kyphosis',palette='Set1')
#The output for this line of code can be viewed in issue section of this repository  - https://tinyurl.com/ybp3992c


#Train test split 
from sklearn.model_selection import train_test_split

X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#Let's train a single decision tree 
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


#Let's evaluate our decision tree
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


#           precision    recall  f1-score   support
#     absent       0.85      0.85      0.85        20
#    present       0.40      0.40      0.40         5
#
#avg / total       0.76      0.76      0.76        25


print(confusion_matrix(y_test,predictions))

# [[17  3]
# [ 3  2]]

#Tree Visualization 
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[1:])
features

#['Age', 'Number', 'Start']

#The output for this line of code can be viewed at issue section of this repository - https://tinyurl.com/ycp8xm8s


#Random Forests
#Now let's compare the decision tree model to a random forest.

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))

# [[18  2]
# [ 3  2]]

print(classification_report(y_test,rfc_pred))
#            precision    recall  f1-score   support
#     absent       0.86      0.90      0.88        20
#    present       0.50      0.40      0.44         5
#
#avg / total       0.79      0.80      0.79        25

