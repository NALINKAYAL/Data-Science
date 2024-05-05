#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# In[2]:


ds=pd.read_csv('SALARY_Datas.csv')
ds.head()


# In[3]:


ds['Purchased'].unique()


# In[4]:


x=ds.iloc[:,[2,3]].values #Age and EstimatedSalary


# In[5]:


y=ds.iloc[:,4].values #Purchased


# In[6]:


y


# In[7]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.25,random_state=0)
print(ytrain)


# In[8]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)
print(xtrain[0:10])


# In[9]:


from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression (random_state=42)
classifier.fit(xtrain, ytrain)
#LogisticRegression(random_state=0)


# In[52]:


#Prediction
y_pred=y_pred= classifier.predict(xtest)
y_pred


# In[53]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(ytest,y_pred)
print("Confusion Matrix: \n",cm)


# In[54]:


from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(ytest, y_pred))


# In[65]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Assuming y_test contains the true labels and y_pred contains the predicted labels by your classifier

# Accuracy
accuracy = accuracy_score(ytest, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(ytest, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(ytest, y_pred)
print("Recall:", recall)


# Confusion Matrix
conf_matrix = confusion_matrix(ytest, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[55]:


from matplotlib.colors import ListedColormap
x_set,y_set = xtest, ytest
x_t, y_t = xtrain, ytrain


# In[57]:


# Function to plot decision boundaries
def plot_decision_boundaries(x, y, title):
    # Creating a mesh grid for Age and Estimated Salary
    x1, x2 = np.meshgrid(np.arange(start=x[:,0].min()-1, stop=x[:,0].max()+1, step=0.01),
                         np.arange(start=x[:,1].min()-1, stop=x[:,1].max()+1, step=0.01))

    # Getting predictions for each point in the mesh grid
    Z = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T)

    # Reshaping predictions to match the shape of the mesh grid
    Z = Z.reshape(x1.shape)

    # Creating a custom colormap for background colors
    cmap_background = ListedColormap(['red', 'green'])

    # Contour plot with background colors
    plt.contourf(x1, x2, Z, alpha=0.3, cmap=cmap_background)

    # Plotting the dataset
    for i, j in enumerate(np.unique(y)):
        plt.scatter(x[y==j, 0], x[y==j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=f'Class {j}')

    # Adding labels and legend
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


# In[58]:



# Plotting decision boundaries for the training set
plot_decision_boundaries(x_t, y_t, 'Classifier (Training set)')

# Plotting decision boundaries for the test set
plot_decision_boundaries(x_set, y_set, 'Classifier (Test set)')


# In[59]:


x=ds.iloc[:,[2,3]].values #Age and EstimatedSalary


# In[60]:


y=ds.iloc[:,4].values #Purchased
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.25,random_state=0)
print(ytrain)


# In[61]:


#standardization
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)
print(xtrain[0:10])


# In[67]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Assuming you have already loaded your dataset and split it into xtrain, xtest, ytrain, and ytest

# Create an instance of Gaussian Naive Bayes Classifier and fit the data
nb_classifier = GaussianNB()
nb_classifier.fit(xtrain, ytrain)

# Plot the decision boundary
_, ax = plt.subplots(figsize=(6, 4))
plt.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain, edgecolors="k", cmap=plt.cm.Paired)

# Generate a grid of points to visualize the decision boundary
x_min, x_max = xtrain[:, 0].min() - 1, xtrain[:, 0].max() + 1
y_min, y_max = xtrain[:, 1].min() - 1, xtrain[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = nb_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Naive Bayes Classifier Decision Boundary")
plt.show()


# In[68]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Assuming you have already loaded your dataset and split it into xtrain, xtest, ytrain, and ytest

# Create an instance of Gaussian Naive Bayes Classifier and fit the data
nb_classifier = GaussianNB()
nb_classifier.fit(xtest, ytest)

# Plot the decision boundary
_, ax = plt.subplots(figsize=(6, 4))
plt.scatter(xtest[:, 0], xtest[:, 1], c=ytest, edgecolors="k", cmap=plt.cm.Paired)

# Generate a grid of points to visualize the decision boundary
x_min, x_max = xtest[:, 0].min() - 1, xtest[:, 0].max() + 1
y_min, y_max = xtest[:, 1].min() - 1, xtest[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = nb_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Naive Bayes Classifier Decision Boundary")
plt.show()


# In[71]:



# Predicting on the test set
y_pred = classifier.predict(xtest)

# Calculating KPIs
accuracy = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred)
recall = recall_score(ytest, y_pred)

conf_matrix = confusion_matrix(ytest, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:




