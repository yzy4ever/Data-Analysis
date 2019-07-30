import pandas as pd
import numpy as np

#导入数据
filename="lesson.csv"
data=pd.read_csv(filename,encoding="gbk")

x=data.iloc[:,1:5].values
y=data.iloc[:,5:6].values

#将数据标准化 1表示积极 0表示消极
for i in range(0,len(x)):
    for j in range(0,len(x[i])):
        if(x[i][j]=="是" or x[i][j]=="高"):
            x[i][j]=1
        else:
            x[i][j]=0

for i in range(0,len(y)):
    if(y[i]=="高"):
        y[i]=1
    else:
        y[i]=0

#将数据类型转换为int型
x=pd.DataFrame(x,dtype="int")
y=pd.DataFrame(y,dtype="int")

#建立模型
from sklearn.tree import DecisionTreeClassifier as DTC
dtc=DTC(criterion="entropy", max_depth=3 ,max_leaf_nodes=10 ,random_state=1,)   #设置最大深度，最大节点数，防止过拟合
dtc.fit(x,y)

#直接预测结果
testdata=np.array([[1,0,1,1],[1,1,0,0],[1,0,1,0]])
result=dtc.predict(testdata)
print(result)   #1011的预测结果为1（销量高） 1100的预测结果为0（销量低） 1010的预测结果为1（销量高）

#可视化决策树
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
with open("tree.dot","w") as file:
    file=export_graphviz(dtc,feature_names=["real_example","lesson_num","discount","material"],out_file=file)
#在命令行输入 dot -Tpng 文件名.dot -o 文件名.png
#右边表示积极，左表示消极
#二分类熵的取值范围是[0,1]，0是非常确定，1是非常不确定