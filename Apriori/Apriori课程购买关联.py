from Apriori.apriori import *
import pandas as pd

filename="C:/应用程序/MyPrograms/Python/Apriori/lesson_buy.xls"
dataframe=pd.read_excel(filename,header=None)   #使得第一行不被作为表头

#数据转换
#将各人购买的课程用10来表示
change=lambda x:pd.Series(1,index=x[pd.notnull(x)])
mapok=map(change,dataframe.as_matrix())   #map(func,[1,2,3,4])   将1，2，3，4分别带入func()中去计算
data=pd.DataFrame(list(mapok)).fillna(0)

#临界支持度、置信度设置
spt=0.2   #支持度大于0.2才会显示   支持度：AB同时的发生的概率
cfd=0.5   #置信度大于0.5才会显示   置信度：A已发生的情况下发生B的概率

#使用apriori算法计算关联结果
find_rule(data,spt,cfd,"-->")

