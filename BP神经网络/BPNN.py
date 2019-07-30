#步骤
#1.读取数据
#2.Sequential   建立模型
#3.Dense   建立层
#4.Activation   激活函数
#5.compile   编译
#6.fit   训练
#7.验证（测试，分类预测）

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation


#数据导入
data=pd.read_csv("lesson.csv",encoding="gbk")

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
x=pd.DataFrame(x,dtype="int").values
y=pd.DataFrame(y,dtype="int").values


model=Sequential()
#输入层
model.add(Dense(10, input_dim=len(x[0])))  #Dense(输入层数,特征数)
model.add(Activation("relu"))   #激活函数 还有softmax tanh sigmoid linear

#输出层
model.add(Dense(1,input_dim=1))
model.add(Activation("sigmoid"))

#编译
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])   #loss:损失函数,optimizer:求解方法,class_mode:指定模式

#训练
model.fit(x,y,nb_epoch=1000,batch_size=100)   #nb_epoch:训练次数,batch_size:p大小（视情况定）

#预测分类
result=model.predict_classes(x).reshape(len(x))