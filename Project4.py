#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as n


# In[15]:


page = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
df1 = pd.read_csv(page, index_col=False, header=None, names=['Poisonous?', 'Cap Color', 'Odor'], usecols=[0,3,5])


# In[16]:


df1.replace(to_replace={'Poisonous?':{'p':1, 'e': 0}}, inplace=True)


# In[17]:


C = pd.Series(df1['Cap Color'])
f = pd.get_dummies(C)

O = pd.Series(df1['Odor'])
g = pd.get_dummies(O)


# In[18]:


new_df = pd.concat([f, g, df1['Poisonous?']], axis=1)
cols = list(new_df.iloc[:, :-1])


# In[19]:


X = new_df.iloc[:, :-1].values
y = new_df.iloc[:, 1].values


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)


# In[ ]:


print(linreg.intercept_)
print(linreg.coef_)


# In[21]:


y_pred = linreg.predict(X_test)


# In[22]:


true = [1, 0]
pred = [1, 0]

print(metrics.mean_absolute_error(true, pred))
print(metrics.mean_squared_error(true, pred))
print(np.sqrt(metrics.mean_squared_error(true, pred)))


# In[23]:


print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[24]:


X = new_df.iloc[:, 11:-1].values

y = new_df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

