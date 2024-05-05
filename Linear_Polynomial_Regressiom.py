#!/usr/bin/env python
# coding: utf-8

# # NALIN KAYAL(211060020)
# 
# # VANSH PANCHAL (211060024)
# 
# # AMEYA JAMKAR (211060005)

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


ds = pd.read_csv('salary_data.csv')


# In[4]:


ds.head()


# In[5]:


ds.dtypes


# In[6]:


x=ds['YearsExperience'] ; y= ds['Salary']


# In[7]:


x=ds.iloc[:,:-1].values ; y = ds.iloc[:,-1].values


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 1/3,random_state=0)


# In[9]:


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()


# In[10]:


lin_reg.fit(x_train,y_train)


# In[11]:


y_pred= lin_reg.predict(x_test)


# In[12]:


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,lin_reg.predict(x_train),color='blue')
plt.title('salary vs experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


# In[13]:


lin_reg.score(x_test,y_test)


# In[14]:


lin_reg.score(x_train,y_train)


# In[15]:


lin_reg.predict([[10]])


# In[16]:


lin_reg.coef_


# In[17]:


lin_reg.intercept_


# In[18]:


from sklearn.linear_model import Ridge,Lasso


# In[19]:


rd=Ridge(alpha=3)
rd.fit(x_train,y_train)
rd.score(x_test,y_test)


# In[22]:


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,rd.predict(x_train),color='blue')
plt.title('salary vs experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# Coefficients of the regression line
coefficients = lin_reg.coef_

# Intercept of the regression line
intercept = lin_reg.intercept_

# Regression line equation
print("Equation of the regression line: y = {:.2f}x + {:.2f}".format(coefficients[0],intercept))


# In[42]:


ls=Lasso(alpha=3)
ls.fit(x_train,y_train)
ls.score(x_test,y_test)


# In[43]:


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,lin_reg.predict(x_train),color='blue')
plt.title('salary vs experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


# In[44]:


dataset = pd.read_csv("Position_Salaries.csv")


# In[45]:


X=dataset.iloc[:,1:2].values  

# for the target we are selecting only the salary column which 
#can be selected using -1 or 2 as the column location in the dataset
y=dataset.iloc[:,2].values   
X


# In[52]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg2=PolynomialFeatures(degree=2)
X_poly=poly_reg2.fit_transform(X) 
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)


# In[53]:


plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg2.fit_transform(X)),color='blue')
plt.title('Truth Or Bluff (Polynomial Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# In[48]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg2=PolynomialFeatures(degree=4)
X_poly=poly_reg2.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)


# In[49]:


plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg2.fit_transform(X)),color='blue')
plt.title('Truth Or Bluff (Polynomial Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# In[ ]:




