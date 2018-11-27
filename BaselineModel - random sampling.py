#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

random.seed(10)


# In[2]:


os.getcwd()
data_path = "/defects/src/data/Jureczko/collated_data/"
os.chdir(os.getcwd() + data_path)


# In[3]:


data = pd.read_csv("ant_merged.csv")
print(data.head())
X = data.loc[:,data.columns!='$<bug']
y = data.loc[:,data.columns=='$<bug']


# In[4]:


print("X dim: ",X.shape)
print("Y dim: ",y.shape)


# In[5]:


projList = os.listdir()
projs = [p.split('_')[0] for p in projList]
print("List of projects:",projs)


# In[6]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# print("X_train dim: ", X_train.shape)
# print("X_test dim: ", X_test.shape)
# print("y_train dim: ", y_train.shape)
# print("y_test dim: ", y_test.shape)


# In[13]:


tbl = [[0]*len(projs) for p in projs]

for i in range(len(projs)):
    print("\nIteration ",i)
    trainData = pd.read_csv(projList[i])
    #print("train data: ", projList[i], projs[i])
    #print(trainData.head())
    X_train = trainData.loc[:,trainData.columns!='$<bug']
    y_train = trainData.loc[:,trainData.columns=='$<bug']
    for j in range(len(projs)):
        if (i != j):
            testData = pd.read_csv(projList[j])
            #print("test data: ", projList[j], projs[j])
            #print(testData.head())
            X_test = testData.loc[:, testData.columns!='$<bug']
            y_test = testData.loc[:, testData.columns=='$<bug']
            clf = RandomForestClassifier(n_estimators=1000, n_jobs=1)
            clf.fit(X_train, y_train.values.ravel())
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            #print("Sum: ",sum(y_pred))
            #print("Confusion Matrix: ",cm)
            #print("True Positive: ", cm[1][1])
            #print("False Positive: ", cm[0][1])
            #print("True Negative: ", cm[0][0])
            #print("False Negative: ", cm[1][0])
            #pd = tp/(tp+fn)
            recall = cm[1][1]/(cm[1][1] + cm[1][0])
            #pf = fp/(fp+tn)
            pf = cm[0][1]/(cm[0][1] + cm[0][0])
            
            g = 2/((1/recall) + (1/(1-pf)))
            
            print("Model trained on ",projs[i] + " " +  projs[j])
            #acc = metrics.accuracy_score(y_test, y_pred)
            print("Project: {}, Accuracy: {}, Precision: {}".format(projs[j],acc,
                                                                   metrics.precision_score(y_test, y_pred)))
            tbl[i][j]=g
            #print("Test Project: {}, G-Score {}".format(projs[j], g))
print(tbl)


# In[21]:


df = pd.DataFrame(tbl)
df.columns=projs
df.insert(0,'projects',projs)
print(df)
#print(os.getcwd())
df.to_csv("..//baseline_gscore.csv",index=False)


# In[ ]:





# ### Training a random forest classifier

# In[15]:


clf = RandomForestClassifier(n_estimators=1000, n_jobs=1)
clf.fit(X_train, y_train)


# In[16]:


y_predictions = clf.predict(X_test)


# In[17]:


print("Accuracy: ",metrics.accuracy_score(y_test, y_predictions))
print("Precision: ",metrics.precision_score(y_test, y_predictions))
print("Recall: ",metrics.recall_score(y_test, y_predictions))
print("F1-score: ",metrics.f1_score(y_test, y_predictions))


# In[ ]:




