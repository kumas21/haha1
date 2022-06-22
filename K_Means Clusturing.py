import numpy as np
import pandas as p
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\gokul\Desktop\New Dataset\Mall_Customers.csv")
print(data)
x=data.iloc[:,[3,4]].values

# Finding the optimal number of clusters using the elbow method
Wcc=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    Wcc.append(kmeans.inertia_)
plt.plot(range(1,11),Wcc)
plt.title('Elbow Method Graph')
plt.xlabel("Number of Clusters")
plt.ylabel("wcc list")
plt.show()
kmeans=KMeans(n_clusters=5,init="k-means++",random_state=42)
label1=kmeans.fit_predict(x)
uniq=np.unique(label1)
for i in uniq:
    plt.scatter(x[label1==i,0],x[label1==i,1],label=("cluster",i))
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='x',color='k',label='centroid')
plt.legend()
plt.show()


