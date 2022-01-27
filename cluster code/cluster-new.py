import random
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
pd.set_option('display.max_columns', None)
 #Show all lines
pd.set_option('display.max_rows', None)

pd.set_option('max_colwidth',100)

data = pd.read_csv('cust_new.csv')
print(data)
# Dropping non-numerical value address
df = data.drop("Address", axis=1)

# Getting the X dataset
X = df.values[:,1:] # Slicing the np array to remove id
X = np.nan_to_num(X)
print(X)
clustering_data = data.iloc[:,[1,2,4]]
wcss=[]
for i in range(1,30):
    km = KMeans(i)
    km.fit(clustering_data)
    wcss.append(km.inertia_) # to measure of how internally coherent clusters are
print(np.array(wcss))
fig, ax = plt.subplots(figsize=(15,7))
ax = plt.plot(range(1,30),wcss, linewidth=2, color="red", marker ="8")
plt.ylabel('WCSS')
plt.xlabel('No. of Clusters (k)')
plt.title('The Elbow Method', fontsize = 20)
plt.show()
# Clustering
kms = KMeans(n_clusters=4, init='k-means++')
print(kms.fit(clustering_data))


clusterNum = 8
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

clusters = clustering_data.copy()
clusters['Cluster_Prediction'] = kms.fit_predict(clustering_data)
print(clusters.head())

df["K-means Cluster"] = labels
print(df.head(5))