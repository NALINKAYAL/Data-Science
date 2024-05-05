#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# NALIN KAYAL (211060020)

# VANSH PANCHAL (211060024)

# AMEYA JAMKAR (211060005)


# In[31]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 


# In[2]:


ds=pd.read_csv('Mall_Customers.csv') 


# In[3]:


ds.head()


# In[4]:


ds.isnull().sum()


# In[37]:


sns.scatterplot(x=ds['Age'], y=ds['Annual Income (k$)'], data=ds)


# In[33]:


sns.scatterplot(x=ds ['Annual Income (k$)'], y=ds['Spending Score (1-100)'], data=ds) 


# In[10]:


x = ds.iloc[:, [3,4]].values 


# In[11]:


from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42) 
y_predict=kmeans.fit_predict(x) 
y_predict 


# In[12]:


x[y_predict==0,0] 


# In[15]:


plt.scatter (x[y_predict==0,0],x[y_predict==0,1],s=50,c='blue', marker='+', label='Cluster 1') 
plt.scatter (x[y_predict==1,0],x[y_predict==1,1],s=50, c='green', marker='o', label='Cluster 2') 
plt.scatter (x[y_predict==2,0],x[y_predict==2,1],s=50,c='red', label='Cluster 3') 
plt.scatter (x[y_predict==3,0],x[y_predict==3,1],s=50,c='cyan', label='Cluster 4') 
plt.scatter (x[y_predict==4,0],x[y_predict ==4,1],s=50, c='magenta', label='Cluster 5') # centroid 
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s = 300, c = 'black',marker='x', label = 'Centroid')
plt.title('Clusters of customers') 
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-1)')
plt.legend()


# In[16]:


from sklearn.cluster import KMean
wcss_list= [] #Initializing the list for the values of WCSS 
kmeans = KMeans(n_clusters=11, init='k-means++', random_state= 42) # assuming 11 clusters 
kmeans.fit(x) 
kmeans.inertia_ 


# In[20]:


for i in range(1, 11): 
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42); 
    kmeans.fit(x); 
    wcss_list.append(kmeans.inertia_); 
plt.plot(range(1, 11), wcss_list) 
plt.title('The Elobw Method Graph') 
plt.xlabel('Number of clusters(k)') 
plt.ylabel('wcss_list') 
plt.show()
    


# In[22]:


# Importing the libraries 
import numpy as nm 
import matplotlib.pyplot as plt 
import pandas as pd 
# Importing the dataset 
dataset = pd.read_csv('Mall_Customers.csv') 
X = dataset.iloc[:, [3, 4]].values 


# In[24]:


#Finding the optimal number of clusters usingthe dendrogram 
import scipy.cluster.hierarchy as shc 
dendro = shc.dendrogram (shc.linkage (x, method="ward")) # other methods are there too 
plt.title("Dendrogrma Plot") 
plt.ylabel("Euclidean Distances") 
plt.xlabel("Customers") 
plt.show() 


# In[26]:


#training the hierarchical model on dataset 
from sklearn.cluster import AgglomerativeClustering 
agg_clustering =AgglomerativeClustering (n_clusters=5, affinity='euclidean', linkage='ward') 
y_pred= agg_clustering.fit_predict(x) 
y_pred 


# In[27]:


plt.scatter (x[y_predict==0,0],x[y_predict==0,1],s=50,c='blue', marker='+',label='Cluster 1') 
plt.scatter (x[y_predict==1,0],x[y_predict==1,1],s=50, c='green', marker='o', label='Cluster 2') 
plt.scatter (x[y_predict==2,0],x[y_predict==2,1],s=50,c='red', label='Cluster 3') 
plt.scatter (x[y_predict==3,0],x[y_predict==3,1],s=50,c='cyan', label='Cluster 4') 
plt.scatter (x[y_predict==4,0],x[y_predict==4,1], s=50, c='magenta', label='Cluster 5') 
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s = 300, c = 'black', marker='x', label = 'Centroid') 
plt.title('Clusters of customers') 
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)') 
plt.legend() 


# In[35]:


sns.pairplot(data=ds)


# In[ ]:




