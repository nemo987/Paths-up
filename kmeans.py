import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import confusion_matrix

dataset= datasets.load_breast_cancer()
x= dataset['data']
y_true= dataset['target']

standardx= StandardScaler().fit_transform(x)


km = KMeans(n_clusters=7, init='k-means++', n_init=10, max_iter=300, random_state=4)
y_cluster = km.fit_predict(standardx)
df = pd.DataFrame({'prediction': y_cluster, 'ground-truth': y_true})
ct = pd.crosstab(df['prediction'], df['ground-truth'])
print(ct) 

y_pred = np.zeros((569,))
y_pred[np.where(y_cluster==0)]= 1
y_pred[np.where(y_cluster==1)]= 0
y_pred[np.where(y_cluster==2)]= 0
y_pred[np.where(y_cluster==3)]= 0
y_pred[np.where(y_cluster==4)]= 1
y_pred[np.where(y_cluster==5)]= 1
y_pred[np.where(y_cluster==6)]= 1
print("Confusion matrix: \n", confusion_matrix(y_true, y_pred))
print("Accuracy score: \n", accuracy_score(y_true, y_pred))
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.scatter(standardx[:,0], standardx[:,1], c=y_true, cmap='jet', edgecolor='None', alpha=0.35)
ax1.set_title('Actual labels')
ax2.scatter(standardx[:,0], standardx[:,1], c=y_pred, cmap='jet', edgecolor='None', alpha=0.35)
ax2.set_title('KMeans clustering results')