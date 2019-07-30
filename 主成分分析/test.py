import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA


X, y = make_blobs(n_samples=10000, n_features=3, centers=[[0,0,0], [1,1,1], [2,2,2],[3,3,3]], cluster_std=[0.1,0.2,0.2,0.2],
                  random_state =9)

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o')
plt.show()

data=pd.DataFrame(X)

###降维###
pca=PCA(2)   #期望降的纬数桥接模式是将抽象部分与它的实现部分分离，使它们都可以独立地变化。
pca.fit(data)
reduct=pca.transform(data)   #降维函数

plt.scatter(reduct[:, 0], reduct[:, 1], c = 'red',marker='o')
plt.show()
