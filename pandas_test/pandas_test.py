import pandas as pd

#Series相当于哈希
a=pd.Series([8,9,2,1])
print(a)  #左边是index 右边是value
b=pd.Series([8,9,2,1],index=["one","two","three","four"])
print(b)

a.sort_values()   #降序排列频数
a.sort_index()   #按index降序排列

#DataFrame相当于二维哈希表
c=pd.DataFrame([[5,2,6],[3,5,1],[9,0,8]])
print(c)
d=pd.DataFrame([[5,2,6],[3,5,1],[9,0,8]],columns=["one","two","three"],index=["four","five","six"])  #column的axis=1;index的axis=0
print(d)

#给DataFrame增加新的列
#c["列名"]=值

#用字典形式创建DataFrame
e=pd.DataFrame({
    "one":4,
    "two":[3,5,9],
    "three":list(str(740))
})
e.values[0][0]   #取出e除去列名行名后的第一行第一列 取一列可以用取一行再转置
e.shape   #e的信息 行数列数
print(e)

d.head()#头部数据默认取前五行，可设置行数
d.tail()#尾部数据默认取后五行，可设置行数

#显示有值的数据的统计情况
print(d.describe())   #count列的元素个数，mean列平均值，std标准差，%分位数 参数include='O' 查看离散变量的describe

print(d.info())   #查看每一列的基本统计情况

#查看某一列的大致分布
d['one'].plot.kde()
#d['two'].plot(kind='kde',xlim=(0,10))

#查看某一列数据的计数情况
d['one'].value_counts()


#转置
print(d.T)

#切片
f=c.iloc[:,0:2].values   #取c的所有行 和 第0列到第1列

#选取数据类型为连续变量的特征
data_select = pd.read_csv('C:/Users/YZY/Desktop/kaggle/test.csv')
numeric_feats = data_select.dtypes[[dtype in ('float64','int64') for dtype in data_select.dtypes]].index