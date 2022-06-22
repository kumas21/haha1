import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Read the Csc file
data = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
# add columns sepal width and sepal length to another dataset
df = data.iloc[:, [0, 1]]
print(df)
# give values to X by giving the values from dataset
X = df.iloc[:, [0, 1]].values
# Scatter plot the input data
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.show()
# Train the model using KNN
nbrs = NearestNeighbors(n_neighbors=3)
# Fit the Model
nbrs.fit(X)
# Anomaly Detection
# distances a indexes of K-Nearest Neighbours from output
distances, indexes = nbrs.kneighbors(X)
# Plot Mean of K-distances
plt.plot(distances.mean(axis=1))
plt.show()
# Mark the outliers and sture to an array
# Visually Determine the cutoff Value
outlier_index = np.where(distances.mean(axis=1) > .15)
print(outlier_index)
# Store the rows corresponding to the outlier values
outlier_values = df.iloc[outlier_index]
print(outlier_values)
# Plt scatter and mark outliers
plt.scatter(df['sepal_length'], df['sepal_width'], s=65)
plt.scatter(outlier_values['sepal_length'], outlier_values['sepal_width'], color='r')
plt.show()
