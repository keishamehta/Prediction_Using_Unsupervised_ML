#!/usr/bin/env python
# coding: utf-8

# # Name- Keisha Mehta

# #### Task: Prediction using Unsupervised ML
# #### From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.

# In[1]:


pip install seaborn


# ## Step1- Importing the required libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns


# ## Step2- Loading the data

# In[3]:


iris_df = pd.read_csv('Iris.csv')


# In[4]:


iris_df = iris_df.set_index('Id')
iris_df.head()


# ## Step3- Reading the data 

# In[5]:


iris_df.head(10) #Displays the first 10 rows


# ## Step4- Data Exploration 

# In[6]:


iris_df.info()


# In[7]:


iris_df.shape #Shape of dataset


# In[8]:


iris_df.describe() #Gives detailed view of dataset


# In[9]:


print('No. of duplicated values :', iris_df.duplicated().sum()) #Check for duplicate values


# In[10]:


iris_df = iris_df.drop_duplicates() #Removing duplicate values
iris_df.head()


# ## Step5- FINDING THE OPTIMUM VALUE OF CLUSTERS

# ### (A) Data Visualization

# In[11]:


sns.pairplot(data=iris_df, hue='Species', height=2.5)
plt.show()


# ### (B) K-means Clustering

# In[12]:


x = iris_df.iloc[:, [0, 1, 2, 3]].values


# In[13]:


from sklearn.cluster import KMeans
wcss = [] #within_cluster_sum_of_squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 500, n_init = 10, random_state = 0) 
    kmeans.fit(x) 
    wcss.append(kmeans.inertia_) 


# ### Plotting the elbow method graph

# In[14]:


plt.plot(range(1, 11), wcss, 'go--', color='blue')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-Cluster sum of squares')
plt.grid()
plt.show()


# #### The required value of the number of the clusters from the above graph is 3 (because 3 onwards, the graph becomes almost constant)

# ## Step6- Applying K-Means Classifier

# In[21]:


#Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 500, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# ## Step7- Plotting the Clusters graph/ Visualizing the Clusters

# In[22]:


color = ['green', 'blue', 'yellow']
labels = ['Iris-versicolour', 'Iris-virginica', 'Iris-setosa']
for i in range(0,3):
    plt.scatter(x[y_kmeans == i, 0], x[y_kmeans==i, 1], s=25, color=color[i], label=labels[i])

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'red', label = 'Centroids')

plt.legend()
plt.grid()
plt.show()

