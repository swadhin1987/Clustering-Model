@author: Swadhin
"""
###Theory

Clustring:
1.No Y vriable
2.Objective:To group/Cluster similar points in such wat that 
similar points group together to form  cluster.
3.Points within the cluster are similar to each other(Homogeneous)& accorss cluster points should be
Dissimilar(clusters should be Heterogenous)
4.Within cluster (intra cluster distance)should be minimum & between cluster distance should be Maximum.
5.In Clusring The Similarity=Nearest ditance=Eucilidein Distance which means
 Measures the Distance Between 2 points(ED)
6.Variables are should be In Numeric in Nature.

ED=square root( (x2-x1)^2+(y2-y1)^2)

Example of Clustering:Costmer Segmentation.

Techniques of clustring
1.Kmeans Clustring----Non Heirrciacal----Numeric Data
2.Hierarical Clustring-----This functining slow
3.Density Based Clustering-----DB scan

1.KMEANS CLUSTERING:
Kmeans Algorithms is also called as LLOYD'S Algorithms
Assumption:We should ware of that how many clusters to make.

K=No of Clusters=Distance=Kmeans

Steps:
1.Intialization:We choose 4 Random points(Centroids)Name as-c1,c2,c3,c4 to create
4 Clusters(K1,K2,K3,K4)in such a way that points closer to a centroid form a clusters.
These Cenroids may not be presentin your Data.

2.ssignment:Based upon Nearst Distance we Assign points of Xi's to the nearst centroids
to form intial Cluster.

3.Recompute Centroid:New Centroids aressignmement Based UponAverage of Each Cluster.

4.Repete the Step-2 of assignment for new centroids.

5.Repeate step 2 and step 3 till the time the movement of points stops,at this point
we can say the algorithms is converged

In Kmeans the min problem is that The Intialization of Centroids Is Randomly pick and each
and every time New Result will come.To overcome this issue Kmeans++ is came.

Kmeans++:
1.Pick 1st Point as centroid Randomly says C1

2.For each point in Data Calculate Distance of that point from centroid,such that
probability of a point to be choosen as 2nd centroid is at Maximum Distance from
the C1 centroid.

3.Make the Centroids as far as Possible.

4.Same Process from Step 2 to 4 above.

##Model

# The aim of this problem is to segment the clients of a wholesale distributor based on their 
# annual spending on diverse product categories, like milk, grocery, region, etc

# reading data:

import pandas as pd
cust_data = pd.read_csv("C:\\Users\\Swadhin\\Desktop\\cust_data.csv")
cust_data.shape

# statistics of the data
x=cust_data.describe()
x=pd.DataFrame(x)

#checking missing values:
cust_data.info()

###Data Preparation For Clustring.

#Here, we see that there is a lot of variation in the magnitude of the data. 
#Variables like Channel and Region have low magnitude whereas 
#variables like Fresh, Milk, Grocery, etc. have a higher magnitude.

#Since K-Means is a distance-based algorithm, 
#this difference of magnitude can create a problem. 
#So let’s first bring all the variables to the same magnitude:

cust_data----is My originl Data
data_scaled---is Scaled Data


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cust_data)


#The magnitude looks similar now..Next, let’s create a kmeans function and fit it on the data:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import KMeans

 Why The Below step is As per assumption we said that we have to know that how
many clusters we need to make.so this below step is needed.   

kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data(In these step we are making the clusters)
y=kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
pred=pd.DataFrame(pred)
pred.columns = ['pred_cluster']   (This is the cluster no-1)
new_ds= pd.concat([cust_data, pred], axis=1)
new_ds['pred_cluster'].value_counts()



## elbow curve----which will tell us How many clusters we should create

# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,10):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)     (inertia=Informations)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# Looking at the above elbow curve, we can choose any number of clusters between 5 to 8. Let’s set the number of clusters as 5 and fit the model:

# k means using 5 clusters and k-means++ initialization
kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

#let’s look at the value count of points in each of the above-formed clusters:

frame = pd.DataFrame(pred)
frame.columns = ['cluster_no']
new_ds= pd.concat([cust_data, frame], axis=1)

-----Cluster size
new_ds['cluster_no'].value_counts()

-----Best Clustring
1.A cluster should not be more than 40% 
2.Dunn index=ration of Inter cluster distance/Intra cluster distance
            =Between cluster distance/within cluster distance

#profiling clusters

----Explore cluster 0:
df1 = new_ds.query('(cluster_no == 0)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)



df1 = new_ds.query('(cluster_no == 1)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)
     




df1 = new_ds.query('(cluster_no == 2)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)
     


df1 = new_ds.query('(cluster_no == 3)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)


df1 = new_ds.query('(cluster_no == 4)')
df1['Channel'].value_counts()
df1['Region'].value_counts()
x=df1.describe()
profiling_df1 = pd.DataFrame(x)

