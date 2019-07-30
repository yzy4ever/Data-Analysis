import jieba
import jieba.analyse
import pandas as pd

data=pd.read_excel("D:/京东商品信息.xls")

name=data.T.values[1]
tag=jieba.analyse.extract_tags(" ".join(name),10)   #" ".join(name)可以把列表转为字符串
print(tag)