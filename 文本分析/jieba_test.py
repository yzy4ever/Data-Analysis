import jieba
sentence="我爱中国共产党"
#精准模式 (默认)
w1=jieba.cut(sentence,cut_all=False)
for i in w1:
    print(i)

#全模式
w2=jieba.cut(sentence,cut_all=True)
for i in w2:
    print(i)

#搜索引擎模式
w3=jieba.cut_for_search(sentence)
for i in w3:
    print(i)

#词性标注
import jieba.posseg
w4=jieba.posseg.cut(sentence)
for i in w4:
    print(i)

'''
a形容词
c连词
d副词
e叹词
f方位词
i成语
m数词
n名词
nr人名
ns地名
nt机构团体
'''

#返回高词频词语
import jieba.analyse
tag=jieba.analyse.extract_tags(sentence,3)   #返回3个 TF/IDF 权重最大的关键词
print(tag)

#返回词语的位置
w5=jieba.tokenize(sentence)   #tokenize(sentence，mode="search")返回搜索引擎下的位置
for i in w5:
    print(i)

'''添加词条词典
jieba.add_word("")
jieba.load_userdict("")   #此方法为暂时性添加词典，想永久添加则更改dict.txt文件
'''



