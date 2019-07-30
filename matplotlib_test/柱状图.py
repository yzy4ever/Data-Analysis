import matplotlib.pylab as plt
import numpy as np

x1 = np.random.normal(0,0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

plt.hist(x1,histtype='stepfilled', alpha=0.3,  bins=40,density=True) #(数据，无边线，透明度，柱子密度，Y轴是否为比例)
plt.hist(x2,histtype='stepfilled', alpha=0.3,  bins=40,density=True)
plt.hist(x3,histtype='stepfilled', alpha=0.3,  bins=40,density=True)

plt.show()