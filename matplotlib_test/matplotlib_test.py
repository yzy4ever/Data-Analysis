import matplotlib.pyplot as plt
import numpy as np

x1=[1,3,4,6,7,9]
y1=[17,13,9,7,4,1]

#折线图，散点图   plot(x轴数据，y轴数据，展现形式)   展现形式:'线的样式线的颜色'
plt.plot(x1, y1, '--c')   #虚线-- 青色cyan
plt.plot(x1, y1, 'om')   #圆点o 品红色magente
plt.title("yzy")   #总标题
plt.xlabel("month")   #x轴名称
plt.ylabel("temperature")   #y轴名称
plt.xlim(0, 10)   #x轴范围
plt.ylim(0, 20)   #y轴范围
plt.show()

'''
=====折线图展现形式=====
-直线
--虚线
-.带点的线
:细小虚线

=====散点图展现形式=====
o圆点
s方形
h六角形
H六角形
x叉型
d菱形
p五角星
'''

#随机
y1=np.random.randint(0, 101, 100)   #普通随机数(最小值，最大值+1，随机数的个数)
y2=np.random.normal(10, 1, 10000)   #正态分布随机数(μ平均数，σ标准差，随机数个数)

#直方图
sty=np.arange(5, 15, 0.5)   #x轴范范围5-15，宽度大小0.5
plt.hist(y2, sty, histtype='stepfilled')   #(数据，样式，取消边线)
plt.show()

#拆分视图
plt.subplot(2, 2, 1)   #当前部分为两行两列情况下的第一部分
#作图
plt.subplot(2, 2, 2)   #当前部分为两行两列情况下的第二部分
#作图
plt.subplot(2, 1, 2)   #当前部分为两行一列情况下的第二部分
#作图
plt.show()