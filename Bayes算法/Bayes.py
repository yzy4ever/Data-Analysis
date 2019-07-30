import numpy as np
class Bayes:
    def __init__(self):
        self.length=-1   #数据特征值的长度
        self.label_propotion=dict()   #训练数据中 各个类别占类别总数的比例
        self.vector_dic=dict()   #按照类别标签存储所有训练数据

    def fit(self,traindata:list,labels:list):
        if (len(traindata) != len(labels)):
            raise ValueError("您输入的训练数据数组跟类别标签数组长度不一致")

        self.length = len(traindata[0])  # 训练数据特征值的长度
        labels_num = len(labels)  # 类别标签的数量
        labels_nor = set(labels)  # 类别标签种类的数量

        for thislabel in labels_nor:
            self.label_propotion[thislabel]=labels.count(thislabel)/labels_num   #求当前类别占类别总数的比例

        for vector, label in zip(traindata, labels):  #一个数据的各个特征相当于一个向量
            if (label not in self.vector_dic):
                self.vector_dic[label] = []
            self.vector_dic[label].append(vector)
        print("训练完成")
        return self


    def go(self,testdata,labels):
        if (self.length == -1):
            raise ValueError("您还没有进行训练，请先训练")

        label_rank_Dict = dict()

        for thislabel in labels:
            p=1   #p值存放概率
            this_label_propotion=self.label_propotion[thislabel]   #当前类别标签的比例
            this_label_vector=self.vector_dic[thislabel]   #取出当前类别标签的所有向量
            this_label_vector_num=len(this_label_vector)   #当前类别标签的个数
            this_label_vector=np.array(this_label_vector).T   #转置 便于计算
            for index in range(0,len(testdata)):
                vector=list(this_label_vector[index])   #取出当前类别所有向量的某一位组形成一个新的向量
                p=p*vector.count(testdata[index])/this_label_vector_num   #计算测试数据某一维占该类标签训练数据对应那一维的比例
            label_rank_Dict[thislabel]=p*this_label_propotion   #最后乘上当前标签的比例
        return sorted(label_rank_Dict,key=lambda x:label_rank_Dict[x],reverse=True)[0]   #返回p值最大的对应标签

