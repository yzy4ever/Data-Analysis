import pandas as pd
from numpy import zeros
from Bayes算法 import Bayes
from os import listdir

def traindata():
    labels = []
    train_file=listdir("C:/应用程序/MyPrograms/Python/Bayes算法/traindata")
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
    test_file = listdir("C:/应用程序/MyPrograms/Python/Bayes算法/testdata")
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

######################################################
all_labels=[0,1,2,3,4,5,6,7,8,9]   #所有标签
train_arr,labels=traindata()
bys=Bayes()
bys.fit(train_arr.tolist(),labels)


test_data = Testdata().tolist()

for i in range(0,len(test_data)):
    print("第("+str(i+1)+"/"+str(len(test_data))+")个数据的识别结果为："+str(bys.go(test_data[i],all_labels)))
