#!/usr/bin/env python
# coding: utf-8

# In[174]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.model_selection import cross_val_score
import time


# In[ ]:





# In[134]:


df = pd.read_csv(r"C:\Users\weizi\Desktop\IE 517\HW6\ccdefault.csv")


# In[135]:


df.head()


# In[136]:


df.describe()


# In[137]:


df.shape


# In[138]:


#Part 1: Random test train splits
#Run in-sample and out-of-sample accuracy scores for 10 different samples by changing 
#random_state from 1 to 10 in sequence. 

X = df.iloc[:, 2:11].values
y = df['DEFAULT'].values

#Using the ccdefault dataset, with 90% for training and 10% for test (stratified sampling)
#and the decision tree model that you did in Module 2:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

Tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
Tree.fit(X_train, y_train)
y_pred = Tree.predict(X_test)
print(accuracy_score(y_test, y_pred))


# In[184]:



steps =[('scaler', StandardScaler()), ('tree', DecisionTreeClassifier())]

pipe = Pipeline(steps)



#1.1 Run in-sample and out-of-sample accuracy scores for 10 different samples by changing random_state from 1 to 10 in sequence. 
IN = []
OUT = []

time0 =time.time()

for i in range (1,11):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10,
                                                        stratify = y,random_state = i)
    pipe.fit(X_train, y_train)
    y_pred_test = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)
    IN.append(accuracy_score(y_train,y_pred_train))
    OUT.append(accuracy_score(y_test,y_pred_test))
    #print('For i = ',i, ':', 'Accuracy scores for In sample: %.4f, Accuracy scores for out sample: %.4f' %
    #      (accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)))
    
    
  


# In[185]:


time1 =time.time()


#1.2 calculate the mean and standard deviation on the set of scores.  
#Report in a table format.


    
    #print('Mean of Accuracy scores for In sample:  %.4f'% np.mean(IN))
    #print('Standard deviation of Accuracy scores for in sample %.4f'% np.std(IN))
    
    #print('Mean of Accuracy scores for out sample: %.4f'% np.mean(OUT))
    #print('Standard deviation of Accuracy scores for out sample: %.4f '% np.std(OUT))
print(time1-time0)


# In[ ]:





# In[180]:


#Part 2: Cross validation
#Now rerun the model using cross_val_scores with k-fold CV (k=10).  
#Report the individual fold accuracy scores, the mean CV score and the standard deviation 
#of the fold scores.  Now run the out-of-sample accuracy score.  Report in a table format.

time2=time.time()
IN_cvscore = cross_val_score(DecisionTreeClassifier(), X_train, y_train, cv=10)


print (IN_cvscore)


# In[181]:


OUT_cvscore = cross_val_score(DecisionTreeClassifier(), X_test, y_test, cv=10)


print ( IN_cvscore )
time3=time.time()

print(time3-time2)


# In[187]:


print("My name is {Zicheng Wei}")
print("My NetID is: {wei41}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




