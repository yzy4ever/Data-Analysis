import pandas as pd
import numpy as np

filename="luqu.csv"
data=pd.read_csv("luqu.csv",encoding="gbk")

x=data.iloc[:,1:4].values

from sklearn.cluster import Birch
from sklearn.cluster import KMeans
kms=KMeans(n_clusters=2,n_jobs=2,max_iter=200)   #K值（分的类数），n_jobs线程数，max_iter最大循环次数
y=kms.fit_predict(x)

data["cluster"]=y
data.to_csv("luqu.csv",index=False)   #将分类的信息写入文档