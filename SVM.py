#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Step 1: Load dataset
iris = datasets.load_iris()  # You can use any other dataset as well
X = iris.data
y = iris.target

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the LinearSVC model
model = LinearSVC(random_state=42)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Test the model
y_pred = model.predict(X_test)

# Step 6: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 7: Print classification report
print("Classification report:")
print(classification_report(y_test, y_pred))


# In[17]:


def plot_decision_boundary(X, y, model):
    # Create a mesh to plot in
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    # Get the legend elements from scatter plot
    handles, labels = scatter.legend_elements()

    # Check that `handles` and `labels` match in length
    if len(handles) != len(iris.target_names):
        print("Mismatch in length of handles and iris.target_names.")
        return

    # Create a legend
    plt.legend(handles=handles, labels=iris.target_names, title="Classes")

    # Set plot limits
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary of Linear SVM')

    plt.show()

# Reduce data to 2D using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(iris.data)

# Split data into train and test sets
X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(X_reduced, iris.target, test_size=0.2, random_state=42)

# Initialize and train the LinearSVC model
model = LinearSVC(random_state=42)
model.fit(X_train_reduced, y_train)

# Plot the decision boundary
plot_decision_boundary(X_test_reduced, y_test, model)


# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Generate a non-linear dataset
X, y = make_regression(n_samples=100, n_features=1, noise=20.0, random_state=42)
# Add non-linearity to the data
y = np.sin(2 * np.pi * X).flatten() * y

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the SVR model with a non-linear kernel (RBF kernel)
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Test the model by making predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Calculate and print performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Step 7: Plot the results
# Sort X_test for better visualization
sorted_indices = np.argsort(X_test, axis=0)
X_test_sorted = X_test[sorted_indices].flatten()
y_pred_sorted = y_pred[sorted_indices]

# Plot the predicted values and actual values
plt.scatter(X_test, y_test, label='Actual Values', color='blue', alpha=0.6)
plt.plot(X_test_sorted, y_pred_sorted, label='SVR Predictions', color='red')
plt.title('SVR for Non-Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


# In[ ]:




