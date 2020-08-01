# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 22:39:53 2020

@author: kisha
"""

'''
Women, especially above 30 years are more susceptible to breast cancer. 
The tumours on breasts may or may not be cancerous. The clinical reports help upto an 
extent in its diagnosis, where mammography stands important in detecting it early. Though 
doctors may find it difficult to detect cancer by the nature of some tumours. The early 
diagnosis of breast cancer can improve the prognosis and chances of survival significantly 
by timely treatment. Also, finding benign tumours will also save the patients from 
undergoing unnecessary treatments. 

Use your machine learning skills to develop a model inorder to predict the nature of 
the tumours.

Comment your doubts.

PS: Your work, especially on Data Analysis (understandings and observations from 
visualization) should be well explained with docstrings (triple quotes multi line 
comments). Work may even be submitted in github and the link to your profile may be 
submitted in classroom as a reponse to this assignment.
'''

#Importing the csv file
import pandas as pd
data=pd.read_csv(r'D:\Study\COVID-19\Machine Learning\DLithe\Datasets\Breast_Cancer.csv')

#Diagnosis: M-Malignant, B-Benign

#Dropping the last unnamed column from the data
data.drop('Unnamed: 32',axis=1,inplace=True)
#Dropping the id column from the data as it is redundant
data.drop('id',axis=1,inplace=True)

'''
The presented data is categorical in nature
So using logistic regression is the proper approach

It's observed that diagnosis is depended on more than 2 variables,
hence find the multivariate feature-feature relation is important.
This can be done using pair plot.
'''
import seaborn as sb
sb.pairplot(data, hue='diagnosis',height=2,markers=['s','D'])
#Array Creation

'''
Note: Since the diagnosis is not in numerical format, it needs to be changed to make it
efficient for data training and testing.
Replacing the strings with integers will help solve this.
Benign-0    (B=0)
Malignant-1 (M=1)
  
'''
classes={'B':0,
         'M':1}

#Replacing the data

data.replace({'diagnosis':classes},inplace=True)

#X=Feature, Y=Target
x=data.iloc[:, data.columns !='diagnosis'].values 
#Note: the columns except 'diagnosis' is selected for x
y=data.iloc[:, 0].values
#Note: the 0th column is the diagnosis column, which is the target

#Splitting Universal Data into train and test sets using train_test_split() method
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.8,random_state=39)

#Selection of Algorithm
#Choosing logistic regression as the data is of classification type

from sklearn.linear_model import LogisticRegression
LogReg=LogisticRegression()

#Training---------------------------------------------
#The fit method of LogReg is used for training the data
LogReg.fit(x_train,y_train)

#Testing----------------------------------------------
'''
The score method is used to find the accuracy of our testing data.
The predict method is used to predict the output of the train data, which can 
then be used to draw the confusion matrix in association with y_test
'''

LogAccuracy=LogReg.score(x_test,y_test)
LogPredict=LogReg.predict(x_test)

#Confusion Matrix
'''
Once the data is predicted and the accuracy is calculated, a confusion matrix is created
to visualize the predicted data in accordance to the test data.
This will help us understand the strength of the model.
'''
#confusion_matrix method is used
from sklearn.metrics import confusion_matrix
conmat = confusion_matrix(y_test,LogPredict) 

































