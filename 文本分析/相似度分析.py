'''
主要步骤
1:读取文档
2:对多篇文档进行分词
3:整理文档成指定格式
4:计算词语频率
5:【可选】对频率低的词语进行过滤
6:通过语料库建立词典
7:加载要对比的文档
8:将要对比的文档通过doc2bow转化为稀疏向量
9:对稀疏向量处理，得到新语料库
10:将新语料库通过tfidfmodel进行处理，得到tfidf
11:通过token2id得到特征数
12:计算系数矩阵相似度，从而建立索引
13:得到最终相似度结果
'''

#本例子为测试哪个doc与dec_test相似度最高
from gensim import corpora,models,similarities
import jieba

doc0 = "我不喜欢上海"
doc1 = "上海是一个好地方"
doc2 = "北京是一个好地方"
doc3 = "上海好吃的在哪里"
doc4 = "上海好玩的在哪里"
doc5 = "上海是好地方"
doc6 = "上海路和上海人"
doc7 = "喜欢小吃"
doc_test="我喜欢上海的小吃"

#将所有doc添加到一个all_doc中
all_doc=[]
all_doc.append(doc0)
all_doc.append(doc1)
all_doc.append(doc2)
all_doc.append(doc3)
all_doc.append(doc4)
all_doc.append(doc5)
all_doc.append(doc6)
all_doc.append(doc7)

#将所有doc进行分词，并保存在all_doc_list中
all_doc_list=[]
for doc in all_doc:
    doc_list=[word for word in jieba.cut(doc,cut_all=False)]
    all_doc_list.append(doc_list)

#将doc_test进行分词
doc_test_list=[word for word in jieba.cut(doc_test,cut_all=False)]

#根据all_doc_list制造语料库
dictionary = corpora.Dictionary(all_doc_list)

print(dictionary.token2id)   #用数字对所有词进行了编号

#将all_doc_list转为二元组向量
corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]   #(词语编号，频次数)

#将doc_test_list转为二元组向量
doc_test_vec = dictionary.doc2bow(doc_test_list)

#使用TF-IDF模型对语料库建模
tfidf = models.TfidfModel(corpus)

print(tfidf[doc_test_vec])   #待测试文本每个词的TF_IDF值

#对每个目标文档，分析测试文档的相似度
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
sim = index[tfidf[doc_test_vec]]
print(sim)

#根据相似度进行排序
print(sorted(enumerate(sim), key=lambda item: -item[1]))   #由结果可知doc7与待测文档相似度最高