import numpy

x=numpy.array(['b','a','0','9'],
              dtype='<U1')

y=numpy.array([[6,2,5],
              [4,0,8],
              [7,1,3]])

z=numpy.array([[0,0,0],
              [0,0,0,]])

print(x,x.dtype)
print(y)

#转列表
print(y.tolist())

#排序
x.sort()   #返回排好的序列
y.sort()

x.argsort()   #返回排名序号

#整合
z1=numpy.concatenate((y,z))   #将z加到y下面
print(z1)

#增加列
numpy.column_stack((x,x))   #将x加到x右边

#列扩展
x2=numpy.tile(x,2)   #把[0,9,a,b]列扩展两次变为[0,9,a,b,0,9,a,b]
print(x2)

#行扩展
x3=numpy.tile(x,(3,1))   #把[0,9,a,b]行扩展三次变为三行[0,9,a,b]
print(x3)


#取最大值最小值
y1=y.max()
y2=y.min()

#求和
y.sum()   #所有数字之和
y.sum(axis=1)   #每一行各列求和


#切片
#数组[起始下标:最终下标+1]
x1=x[1:3]
x2=x[:]

#随机
numpy.random.randint(0,101,100)   #普通随机数(最小值，最大值+1，随机数的个数)
numpy.random.normal(10,1,10000)   #正态分布随机数(μ平均数，σ标准差，随机数个数)



