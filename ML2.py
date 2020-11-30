#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


import numpy as np


# In[5]:


data = pd.read_csv(r"C:\Users\Dharmendra\Desktop\bk1.csv")


# In[6]:


data.head()


# In[7]:


df['SYMBOL']


# In[8]:


data['SYMBOL']


# In[9]:


data['TOTALTRADES']


# In[10]:


data[['SYMBOL','TOTALTRADES']]


# In[13]:


data['TOTALTRADES' >= 1000]


# In[24]:


TT = ['TOTALTRADES']
df = data(TT,columns=['group'])
df.loc[df['group'] <= 10000, 'Volatile'] ='True'
df.loc[df['group'] > 10000,'Volatile'] = 'False'
print(df)


# In[21]:


cols = ['TOTALTRADES','SYMBOL']
data.query('TOTALTRADES > 10000')[cols]


# In[23]:


cols = ['TOTALTRADES','SYMBOL']
data.query('TOTALTRADES > 100000')[cols]


# In[25]:


rows = ['OPEN','CLOSE','HIGH','LOW']
data.query('HIGH > OPEN')[rows]


# In[59]:


data['range'] = (data['HIGH'] - data['OPEN'])
print(data)


# In[41]:


rows = ['OPEN','CLOSE','HIGH','LOW']
data.query('HIGH - OPEN')[rows]


# In[48]:



print(rows)


# In[51]:


rows = ['OPEN','CLOSE','HIGH','LOW']
data.query('OPEN - HIGH')[rows]


# In[52]:


print(data)


# In[61]:


pot = ['SYMBOL', 'range']
data.query('range > 100')[pot]


# In[66]:


from pandas import DataFrame


# In[69]:


data[data.TOTALTRADES > 100]


# In[72]:


data['VOLATILE'] = (data['TOTALTRADES'] >100)
print(data)


# In[ ]:




