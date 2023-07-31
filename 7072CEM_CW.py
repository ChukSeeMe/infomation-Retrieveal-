#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imblearn
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


# In[ ]:





# Source: 
# MAGIC Gamma Telescope Data Set
# ## https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope 

# In[3]:


cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "FM3Trans", "FAlpha", "fDist","class"]
df = pd.read_csv("magic04.data", names = cols)
df.head()


# In[4]:


df["class"].unique()


# ***CLASSIFICATION***: Here, we try to understand the dataset.
# Having known the different features in the dataset, i tried to classify each entry using "g" for Gamma and "h" for Hydron. 
# With this, i will be able to predict the future occurance as regards expected rays. 

# In[5]:


#processing my dataset and converting the class variable to integer.
#This will get the algorithms to view the Gamma and Hydron string as integer
df["class"] = (df["class"] == "g").astype(int)


# In[6]:


df.head()


# In[18]:


#Here i checked the class involved in my dataset and found them to be integers (1) and (0)
#Here 1 represents"Gamma" rays while the 0 represents the Hadron rays if the patient has hepatitis and 2 represents if they do not have hepatitis 
df["class"].unique()


# In[19]:


#Drawing a histogram of the dataset to help me understand more about the features of the dataset and overview understanding of the dataset
#measuring the distribution of the features

for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label], color='blue', label = 'Gamma', alpha = 0.7, density = True) #gamma
    plt.hist(df[df["class"]==0][label], color='red', label = 'Hadron', alpha = 0.7, density = True) #hydron
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()


# Segmenting my dataset into: Training, Testing and Validation Dataset

# In[20]:


train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])


# In[21]:


print(len(train[train["class"]==1]))
print(len(train[train["class"]==0]))
#This shows the distribution of the hadron and gamma particles as seen in the dataset
df.head()
df.describe()
df['class'].value_counts().plot(kind='bar')
plt.show()


# In[22]:


#Scaling the dataset
#I had to scale the dataset to be able to be equal to the mean of the standard deviation of the labels 
def scale_dataset(dataframe, oversample=False):
  x = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values
# After printing the length of my training dataset, i found out that the samples are not equal.
#a that, i decided to oversample just to balance it up.
  scaler = StandardScaler()
  x = scaler.fit_transform(x)
  if oversample:
    ros = RandomOverSampler() 
    x, y = ros.fit_resample(x, y)
  #Take from the less class of the dataset and keep sampling till they all match
#creating the data into a two dimension array, having noticed that my y label is one dimension and x label is two dimension
  data = np.hstack((x, np.reshape(y, (-1, 1)))) 

  return data, x, y 


# In[23]:


train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample=False)

#By Oversampling, i made the length of the two class feature to be equal  to eachother 


# In[24]:


sum(y_train ==1)


# In[25]:


sum(y_train ==0)


# In[26]:


import matplotlib.pyplot as plt

# Function to plot histograms for each dataset
def plot_histogram(y_data, dataset_name):
    plt.figure(figsize=(10, 6))
    plt.hist(y_data, bins=np.arange(y_data.min(), y_data.max() + 2) - 0.5, edgecolor='black')
    plt.xlabel('Class')
    plt.ylabel('probability')
    plt.title(f'Distribution of y values for MAGIC DATASET Data')
    plt.xticks(np.unique(y_data))
    plt.show()

# Plot histograms for training, validation, and testing datasets
plot_histogram(y_train, 'Training')


# In[27]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y


# **LOGISTICS REGRESSION** 

# In[28]:




from sklearn.linear_model import LogisticRegression

lg_model = GaussianNB()
lg_model = lg_model.fit(x_train, y_train)
y_pred = lg_model.predict(x_test) #Making my prediction using the testing data model 
print(classification_report(y_test, y_pred))

#Result: It can be noticed that even though the Accuracy is 72%, giving us a better clarification 
# as compared to the naive bayes algorithm, that the best so far that has given a more precise sccuracy is the KNN

# Scale the dataset and oversample if needed
data, x, y = scale_dataset(df, oversample=True)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a classifier
clf = LogisticRegression()
clf.fit(x_train, y_train)

# Calculate the predicted probabilities
y_pred_prob = clf.predict_proba(x_test)[:, 1]

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
plot_confusion_matrix(clf, x_test, y_test, ax=ax, cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix')
plt.show()


# ***DECISIONTREE**

# In[29]:




from sklearn.metrics import roc_curve, auc, confusion_matrix, plot_confusion_matrix
import seaborn as sns


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Create a decision tree model
dt = DecisionTreeClassifier()

# Train the model
dt.fit(x_train, y_train)

# Predict the target variable
y_pred = dt.predict(x_test)  # Use x_test instead of y_test

print(classification_report(y_test, y_pred))

plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, rounded=True)
plt.show()



# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate the area under the curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using Seaborn's heatmap
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Alternatively, you can use the plot_confusion_matrix function from sklearn
plot_confusion_matrix(dt, x_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# ***K NEAREST NEIGHBORS***

# In[30]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
knn_model = KNeighborsClassifier(n_neighbors=5) #First step i took was to classify the dataset
#by telling the model how many features it will take for the classification
#Here i used 5 features for the classification to check if they fit into the model 
knn_model.fit(x_train, y_train)


# In[31]:


y_pred = knn_model.predict(x_test)
y_pred #Just to clarify that my testing and prediction features tallies, i checked for y_test and y_prediction
#i found out that both of them are the same.


# In[32]:


y_test


# In[33]:


print(classification_report(y_test, y_pred))



#In the chat below, the accrucacy of the points is 80%
#In the precision, i found out that the precision for both class "0" and "1" are 65% and 74% respectively
#The recall is at 41 and 89 making it good as the recall show the possibility of getting a specific class in the dataset. 
#Having analyzed this data, it can be seen that the possibility of getting a hydron was more than getting gamma rays.


# In[34]:


from sklearn.metrics import roc_curve, auc, confusion_matrix, plot_confusion_matrix
import seaborn as sns

# Train the model
knn_model.fit(x_train, y_train)

# Predict the target variable
y_pred = knn_model.predict(x_test)

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate the area under the curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using Seaborn's heatmap
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Alternatively, you can use the plot_confusion_matrix function from sklearn
plot_confusion_matrix(knn_model, x_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# ***K MEANS CLUSTERING***

# In[40]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(x_test)
cluster_labels = kmeans.labels_

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_test)


# Create a k-means model
kmeans = KMeans(n_clusters=2)

# Train the model
kmeans.fit(x_test)

# Predict the clusters
y_pred = kmeans.predict(x_test)


# Visualize the clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.title('KMeans Clustering with PCA Visualization')
plt.show()


# ***NAIVE BAYES***

# In[42]:


pip install my_library


# In[35]:


from sklearn.naive_bayes import GaussianNB


# In[36]:


nb_model = GaussianNB()
nb_model.fit(x_train, y_train) #Here i am trying to fit in my training set into the data model 


# In[39]:


y_pred = nb_model.predict(x_test) #Making my prediction using the testing data model 
print(classification_report(y_test, y_pred))

#Result: Here the prediction dataset shows that the accuracy is at 72% 
#Precision shows that the gamma rays has more precision of 74% as compared to the hydron ray at 64%



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
#from my_library import nb_model


# Fit the model
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
y_pred = nb_model.predict(x_test)

# Print classification report
print(classification_report(y_test, y_pred))


# Plot ROC curve
y_prob = nb_model.predict_proba(x_test)[:, 1]  # Probability of positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ***SUPPORT MACHINE VECTOR***

# In[41]:


from sklearn.svm import SVC

# Create an SVM model
svm = SVC()

# Train the model
svm.fit(x_test, y_test)

# Predict the target variable
y_pred = svm.predict(x_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)


# In[ ]:




