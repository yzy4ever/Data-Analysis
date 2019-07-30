from numpy import *
import operator
from os import listdir
from PIL import Image

def knn(k,testdata,traindata,labels):
    traindata_size=traindata.shape[0]   #一行一个traindata，取行数即个数
    dif=tile(testdata,(traindata_size,1))-traindata   #将待测数据扩展为与训练数据相同的维数，并求差
    sq_dif=dif**2   #差的平方
    sum_sq_dif=sq_dif.sum(axis=1)   #每一行的各列求和
    distance=sum_sq_dif**0.5   #开方，欧几里德距离
    sort_distance=distance.argsort()
    count={}   #建立一个字典
    for i in range(0,k):
        vote=labels[sort_distance[i]]  #取得距离最小行的lable
        count[vote]=count.get(vote,0)+1
    sort_count=sorted(count.items(),key=operator.itemgetter(1),reverse=True)
    return sort_count[0][0]

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

def traindata():
    labels = []
    train_file=listdir("C:/应用程序/MyPrograms/Python/KNN算法/traindata")
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

def img_to_str(fname):
    fh = open(fname.split(".")[0]+".txt", "w+")
    im = Image.open(fname)
    width = im.size[0]  # 图片的宽
    height = im.size[1]  # 图片的高

    for i in range(0, width):
        for j in range(0, height):
            cl = im.getpixel((j, i))
            if ((cl[0] + cl[1] + cl[2]) < 20):  # 如果rgb为000即黑色  <20 灰色宽容度
                fh.write('1')
            else:
                fh.write('0')
        fh.write('\n')
    fh.close()


train_arr,labels=traindata()
test_file="test.jpg"
img_to_str(test_file)
test_arr=data_to_array(test_file.split(".")[0]+".txt")
result=knn(3,test_arr,train_arr,labels)
print("识别结果："+str(result))
