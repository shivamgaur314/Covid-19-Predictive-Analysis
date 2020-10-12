#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import csv


# In[4]:


data= pd.read_csv(r"C:\Users\Shivangi\Desktop\data\covid_19_india.csv")


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.drop(['ConfirmedIndianNational','ConfirmedForeignNational'],axis='columns',inplace=True)


# In[8]:


data.head()


# In[9]:


data1=data[data['State/UnionTerritory']=='Maharashtra']


# In[10]:


data1.head()


# In[11]:


data1.tail()


# In[12]:


plt.scatter(data1.Confirmed,data1.Cured)
plt.show()


# In[52]:


x=data1.Confirmed


# In[53]:


y=data1.Deaths


# In[54]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


# In[55]:


from sklearn.linear_model import LinearRegression


# In[56]:


lr=LinearRegression()


# In[57]:


lr.fit(x_train.values.reshape(-1,1),y_train.values.reshape(-1,1))


# In[60]:


pred=lr.predict(x_test.values.reshape(-1,1))
pred


# In[64]:


y_test


# In[63]:


lr.score( x_test.values.reshape(-1,1),y_test.values.reshape(-1,1))

