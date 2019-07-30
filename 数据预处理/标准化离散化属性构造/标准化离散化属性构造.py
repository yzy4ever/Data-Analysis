import pandas as pd
import numpy as np
import pymysql

conn=pymysql.connect(host="127.0.0.1",user="root",password="y1998812",db="mytest")
sql='select price,comment from taobao'
data=pd.read_sql(sql,conn)

###离差标准化：消除刚量（单位）影响以及变异大小因素的影响
data1=(data-data.min())/(data.max()-data.min())
print(data1)

###标准差标准化：消除单位影响以及变量自身变异影响
data2=(data-data.mean())/data.std()
print(data2)

###小数规范化：消除单位影响
k=np.ceil(np.log10(data.abs().max()))   #ceil()进一取整
data3=data/10**k
print(data3)

###等宽离散化
data4=data[u"price"].copy()
data4=data4.T
data4=data4.values

'''假设划分为等宽的4份，每份对应标签为便宜适中小贵超贵'''
#data4=pd.cut(data4,4,labels=["便宜","适中","小贵","超贵"])
'''非等宽离散化 0-20便宜 20-80适中 80-200小贵 200+超贵'''
data4=pd.cut(data4,[0,20,80,200,data4.max()],labels=["便宜","适中","小贵","超贵"])
print(data4)

###属性构造   为数据添加一列新的属性
ch=data[u"comment"]/data["price"]
data[u"评论价格比"]=ch   #直接为新属性赋值
file="./评论价格比.xls"
data.to_excel(file,index=False)