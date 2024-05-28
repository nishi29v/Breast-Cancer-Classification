#!/usr/bin/env python
# coding: utf-8

# By: Nishita Vaid
# 
# Date: 30 August 2023 - 14 September 2023
# 
# 
# ## Breast Cancer Classification:
# 
# Problem Statement - 
# Classifying Cancer as malignant (M) or benign (B) based on information like radius of tumor, smoothness, 
# compactness, texture, perimeter, etc.
# 
# More about dataset:
# 
#     Target: Cancer type M or B
#     
#     Features: Features: 30 features provided, namely, radius mean, texture mean, perimeter mean, area mean,           smoothness mean, compactness mean, concaviy mean, concave points mean, symmetry mean, fractal dimension mean, radius se, texture se, perimeter se, area se, smoothness se, compactness se, concavity se, concave points se, symmetry se, fractal dimension se, radius worst, texture worst, perimeter worst, area worst, smoothness worst, compactness worst, concavity worst, concave points worst, symmetry worst, fractal dimension worst.
#     
#     Size of the dataset: 569

# In[1]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Data Preprocessing

# In[2]:


# Importing data
data = pd.read_csv("/home/nishita/Documents/Semester3/DC/Cancer_Data.csv")

# Read data (first 10 lines)
data.head(10)


# In[3]:


# More about data
print(f"size of the data: \n{data.shape}")


# In[4]:


print(f"More info about the data: \n{data.info}")


# In[5]:


print(data.dtypes)


# In[6]:


# Checking for nan values
nan_count = data.isna().sum()
print(nan_count)


# In[7]:


# No NaN Values in the dataset.
# Removing the last id column
data = data.drop(['id'], axis = 1)

data.head(10)


# In[8]:


data.shape


# # Data Visualization

# In[9]:


# Now our data is cleaned

# Data Visualization
# Pair plot
columns_to_include = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

sns.pairplot(data[columns_to_include])
plt.show()


# In[10]:


# Scatter Plot between radius mean and texture mean
sns.scatterplot(x = data['radius_mean'], y = data['texture_mean'], c ="Red")
plt.title("Scatter Plot")
plt.xlabel("Radius_Mean")
plt.ylabel("Texture_Mean")
plt.show


# In[11]:


# lmplot
sns.lmplot(x = 'radius_mean', y = 'texture_mean', data = data)
plt.title("Scatter Plot")
plt.xlabel("Radius_Mean")
plt.ylabel("Texture_Mean")
plt.show


# In[12]:


# heatmap

correlation_matrix = data.corr()

sns.heatmap(correlation_matrix, cmap = 'coolwarm')


# # Model Fitting

# In[13]:


# Creating feature and target variable only few features
columns_to_include = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
X = data.iloc[:,1:11]
print(f"Features: \n {X.head()}")
print("\n")
X.shape


# In[14]:


# Target
y = data['diagnosis']
print(f"Target:\n {y.head()}")
print("\n")
y.shape


# In[15]:


# Splitting data in train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)


# In[16]:


# Fitting Logistic Regression Model
from sklearn.linear_model import LogisticRegression
# Define Model
model = LogisticRegression()

# Fitting Model
model.fit(X_train, y_train)


# In[17]:


# Prediction
y_pred = model.predict(X_test)


# In[18]:


# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))  
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[19]:


# Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(f"Logistic Regression Model Accuracy: {accuracy}")


# In[20]:


# Improving accuracy using hyperparameter
from sklearn.model_selection import GridSearchCV

# Define hyperparameters to search 
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'saga'],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')

# Fit the model with the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_


# In[21]:


# Logistic Regression after hyperparameter
best_lr = LogisticRegression(**best_params)
best_lr.fit(X_train, y_train)


# In[22]:


# Evaluate improved model
# Predicting
y_pred_hyper = best_lr.predict(X_test)


# In[23]:


# Confusion Matrix
cm_hyper = confusion_matrix(y_test, y_pred_hyper)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))  
sns.heatmap(cm_hyper, annot=True, fmt="d", cmap="Oranges",
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[24]:


# Accuracy
accuracy_hyper = accuracy_score(y_test, y_pred_hyper)

print(f"Improved Logistic Regression Model Accuracy: {accuracy_hyper}")


# In[25]:


# Applying KNN Model
from sklearn.neighbors import KNeighborsClassifier

# Defining Classifier with k = 3
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Fitting KNN
knn_classifier.fit(X_train, y_train)


# In[26]:


# Prediction
y_pred_knn = knn_classifier.predict(X_test)


# In[27]:


# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))  
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[28]:


# Model Accuracy
accuracy1 = accuracy_score(y_test, y_pred_knn)

print(f"KNN Model with k = 3 Accuracy: {accuracy1}")


# In[29]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Craeting Model
model_dt = DecisionTreeClassifier(random_state=42)

# Model fitting
model_dt.fit(X_train, y_train)


# In[30]:


# Prediction Making
y_pred_dt = model_dt.predict(X_test)


# In[31]:


# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))  
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Reds",
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[32]:


# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Accuracy of Decision tree classifier model: {accuracy_dt}")


# In[33]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Model
model_dt = DecisionTreeClassifier(random_state=42)

# Define hyperparameters to search
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV object
grid_search_dt = GridSearchCV(model_dt, param_grid_dt, cv=5, scoring='accuracy')

# Fit the model with the best hyperparameters
grid_search_dt.fit(X_train, y_train)

# Get the best hyperparameters
best_params_dt = grid_search_dt.best_params_


# In[34]:


# Decision Tree after hyperparameter
best_dt = DecisionTreeClassifier(**best_params_dt)
best_dt.fit(X_train, y_train)


# In[35]:


# Evaluate improved model - decision tree
# Predicting
y_pred_hyper_dt = best_dt.predict(X_test)


# In[36]:


# Confusion Matrix
cm_dt_hyper = confusion_matrix(y_test, y_pred_hyper_dt)

# Plotting confusion matrix
plt.figure(figsize = (8, 6))  
sns.heatmap(cm_dt_hyper, annot = True, fmt = "d", cmap = "Blues",
            xticklabels = ['Predicted Negative', 'Predicted Positive'],
            yticklabels = ['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[37]:


# Evaluate the model
accuracy_dt_hyper = accuracy_score(y_test, y_pred_hyper_dt)
print(f"Accuracy of Decision tree classifier model after applying hyperparameter: {accuracy_dt_hyper}")


# In[38]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Model
model_rf = RandomForestClassifier(random_state = 42)

# Fitting model with training data
model_rf.fit(X_train, y_train)


# In[39]:


# Prediction Making
y_pred_rf = model_rf.predict(X_test)


# In[40]:


# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Plotting confusion matrix
plt.figure(figsize = (8, 6))  
sns.heatmap(cm_rf, annot = True, fmt = "d", cmap = "Blues",
            xticklabels = ['Predicted Negative', 'Predicted Positive'],
            yticklabels = ['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[41]:


# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy of Random Forest: {accuracy_rf}")


# In[42]:


# Random Forest using hyperparameters
# Model 
model_rf = RandomForestClassifier(random_state = 42)

# Define the hyperparameters grid to search
param_grid_rf = {
    'n_estimators': [100, 200, 300],  
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create GridSearchCV instance
grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv = 5, scoring = 'accuracy')

# Fit the model with training data
grid_search_rf.fit(X_train, y_train)


# In[43]:


# Best model - Random forest with hyperparametrs
best_params_rf = grid_search_rf.best_params_
best_estimator_rf = grid_search_rf.best_estimator_


# In[44]:


# Make predictions
y_pred_rf_hyper = best_estimator_rf.predict(X_test)


# In[45]:


# Confusion Matrix
cm_rf_hyper = confusion_matrix(y_test, y_pred_rf_hyper)

# Plotting confusion matrix
plt.figure(figsize = (8, 6))  
sns.heatmap(cm_rf_hyper, annot = True, fmt = "d", cmap = "Reds",
            xticklabels = ['Predicted Negative', 'Predicted Positive'],
            yticklabels = ['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[46]:


# Evaluate the model
accuracy_rf_hyper = accuracy_score(y_test, y_pred_rf_hyper)
print(f"Accuracy of the Random Forest Classifier using hyperparameters: {accuracy_rf_hyper}")


# In[47]:


# SVM
from sklearn.svm import SVC

# model
svm_classifier = SVC()

# Fitting model
svm_classifier.fit(X_train, y_train)


# In[48]:


# Make predictions
y_pred_svm = svm_classifier.predict(X_test)


# In[49]:


# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)

# Plotting confusion matrix
plt.figure(figsize = (8, 6))  
sns.heatmap(cm_svm, annot = True, fmt = "d", cmap = "Greens",
            xticklabels = ['Predicted Negative', 'Predicted Positive'],
            yticklabels = ['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[50]:


# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy of the SVM Classifier: {accuracy_svm}")


# Accuracy score of different ML Classifiers:
# 
# 1. Logistic Regression Model = 90.35%
# 
# 2. Logistic Regression using hyperparameter = 94.73%
# 
# 3. KNN Model = 92.98%
# 
# 4. Decision Tree = 91.22%
# 
# 5. Decision Tree after using hyperparameters = 92.10%
# 
# 6. Random Forest = 92.10%
# 
# 7. Random Forest after applying hyperparameters = 91.22%
# 
# 8. SVM = 90.35%

# # PCA

# In[51]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[52]:


# Loading data
data_pca = data

# Features (X) and target (y)
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']


# In[53]:


# Standardize the features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# In[54]:


# Applying PCA

# components:
components = 2
# PCA
pca = PCA(n_components = components)
X_pca = pca.fit_transform(X_scaled)


# In[55]:


# Plotting scatter plot
# Customizing color
custom_colors = {'M': 'blue', 'B': 'green'}
color = [custom_colors[label] for label in y]

# Plotting
plt.figure(figsize=(8, 6))
plot = plt.scatter(X_pca[:, 0], X_pca[:, 1], c = color, cmap='viridis', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: 2D Scatter Plot')

plt.show()


# In[56]:


# Applying Logistic Regression after PCA
# Splitting data in train and test data
Xpca_train, Xpca_test, ypca_train, ypca_test = train_test_split(X_pca, y, test_size = 0.2, random_state = 45)


# In[57]:


# Model defined
model1 = LogisticRegression()

# Model Fitting
model1.fit(Xpca_train, ypca_train)


# In[58]:


# Prediction
y_pred_pca = model1.predict(Xpca_test)


# In[59]:


# Confusion Matrix
cm_pca = confusion_matrix(ypca_test, y_pred_pca)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))  
sns.heatmap(cm_pca, annot=True, fmt="d", cmap="Reds",
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[60]:


# Model Accuracy
accuracy3 = accuracy_score(ypca_test, y_pred_pca)

print(f"Logistic Regression Model Accuracy after PCA: {accuracy3}")


# In[61]:


# Applying KNN Model after pca

# Defining Classifier with k = 3
k = 3
knn_classifier_pca = KNeighborsClassifier(n_neighbors=k)

# Fitting KNN
knn_classifier_pca.fit(Xpca_train, ypca_train)


# In[62]:


# Prediction
y_pred_knn_pca = knn_classifier_pca.predict(Xpca_test)


# In[63]:


# Confusion Matrix
cm_knn_pca = confusion_matrix(y_test, y_pred_knn_pca)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))  
sns.heatmap(cm_knn_pca, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[64]:


# Model Accuracy
accuracy_knn_pca = accuracy_score(ypca_test, y_pred_knn_pca)

print(f"KNN Model with k = 3 Accuracy after pca: {accuracy_knn_pca}")


# In[65]:


# Decision Tree after pca

# Craeting Model
model_dt_pca = DecisionTreeClassifier(random_state=42)

# Model fitting
model_dt_pca.fit(Xpca_train, ypca_train)


# In[66]:


# Prediction
y_pred_dt_pca = model_dt_pca.predict(Xpca_test)


# In[67]:


# Confusion Matrix
cm_dt_pca = confusion_matrix(ypca_test, y_pred_dt_pca)

# Plotting confusion matrix
plt.figure(figsize = (8, 6))  
sns.heatmap(cm_dt_pca, annot = True, fmt = "d", cmap = "Reds",
            xticklabels = ['Predicted Negative', 'Predicted Positive'],
            yticklabels = ['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[69]:


# Model Accuracy
accuracy_dt_pca = accuracy_score(ypca_test, y_pred_dt_pca)

print(f"Decision tree with pca: {accuracy_dt_pca}")


# Accuracy before PCA for ML classifiers:
# 
# 1. Logistic Regression Model = 90.35%
# 
# 2. Logistic Regression using hyperparameter = 94.73%
# 
# 3. KNN Model = 92.98%
# 
# 4. Decision Tree = 91.22%
# 
# 5. Decision Tree after using hyperparameters = 92.10%
# 
# 6. Random Forest = 92.10%
# 
# 7. Random Forest after applying hyperparameters = 91.22%
# 
# 8. SVM = 90.35%
# 
# After PCA:
# 
# 1. Logistic Regression = 97.36%
# 
# 2. KNN = 92.98%
# 
# 3. Decision Tree = 89.47%
