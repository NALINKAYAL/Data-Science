#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as ss


# In[5]:


df = pd.read_csv('tips.csv')


# In[6]:


df.head()


# In[9]:


df.columns


# In[10]:


df.tail()


# In[12]:


df.describe()


# In[16]:


ss.boxplot(df["total_bill"]).set_title("Box Plot")


# In[19]:


ss.boxplot(x=df["tip"],y=df["smoker"])


# In[20]:


ss.boxplot(x=df["tip"],y=df["day"])


# In[25]:


ss.distplot(df["total_bill"])
ss.distplot(df["tip"])


# In[26]:


ss.boxplot(x=df["tip"],y=df["sex"])


# In[30]:


ss.scatterplot(x='total_bill',y='tip',data=df)


# In[31]:


ss.violinplot(x='total_bill',y='tip',data=df)


# In[36]:


ss.lmplot(x='total_bill',y='tip',data=df)


# In[ ]:




