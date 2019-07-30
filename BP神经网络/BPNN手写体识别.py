import pandas as pd
import numpy as np
from os import listdir
from numpy import zeros

def traindata():
    labels = []
    train_file=listdir("C:/应用程序/MyPrograms/Python/BP神经网络/traindata")
    num=len(train_file)
    # 长度32*32=1024（列），每一行存储一个文件
    # 用一个数组存储所有训练数据，行：文件总数，列：1024
    train_arr = zeros((num, 1024))
    for i in range(0, num):
        this_name = train_file[i]
        this_label = sep_label(this_name)
        labels.append(this_label)
        train_arr[i, :] = data_to_array("traindata/" + this_name)
    return train_arr, labels


def Testdata():
    test_file = listdir("C:/应用程序/MyPrograms/Python/BP神经网络/testdata")
    num = len(test_file)
    test_arr = zeros((num, 1024))
    for i in range(0, num):
        this_name = test_file[i]
        test_arr[i, :] = data_to_array("testdata/" + this_name)
    return test_arr


def sep_label(fname):   #标签为文件名称的第一个首字符
    return int(fname[0])

def data_to_array(fname):   #将32*32的字符串读为1024的一行
    arr=[]
    fh=open(fname)
    for i in range(0,32):
        this_line=fh.readline()
        for j in range(0,32):
            arr.append(int(this_line[j]))
    return arr

#获取数据
train_arr,labels=traindata()

#将labels转为10串表示
labels=pd.DataFrame(labels)
change=lambda x:pd.Series(1,index=x[pd.notnull(x)])
mapok=map(change,labels.as_matrix())
labels_data=pd.DataFrame(list(mapok)).fillna(0)


from keras.models import Sequential
from keras.layers import Dense,Activation

model=Sequential()
#输入层
model.add(Dense(10, input_dim=len(train_arr[0])))  #Dense(输入层数,特征数)
model.add(Activation("relu"))   #激活函数

#输出层
model.add(Dense(10,input_dim=10))   #输出层层数与特征数相同;结果为10位表示，因此特征数为10
model.add(Activation("softmax"))   #softmax适合于多分类

#编译
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])   #loss:损失函数,optimizer:求解方法,class_mode:指定模式

#训练
model.fit(train_arr,labels_data,nb_epoch=10000,batch_size=6)   #nb_epoch:训练次数,batch_size:p大小（视情况定）

#预测结果
test_data = Testdata()
for i in range(0,len(test_data)):
    print("第("+str(i+1)+"/"+str(len(test_data))+")个数据的识别结果为："+str(model.predict_classes(np.array([test_data[i]])).reshape(1)))

#保存模型
model.save("C:/应用程序/MyPrograms/Python/BP神经网络/myModel.h5")

#导入模型
import keras
newModel=keras.models.load_model("C:/应用程序/MyPrograms/Python/BP神经网络/myModel.h5")
