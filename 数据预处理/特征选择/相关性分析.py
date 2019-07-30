'''
相关性分析 pd.corr()
贪婪特征选择法
Lasso回归
'''
def data_clean():
    import pandas as pd

    # 读取数据
    data = pd.read_excel("红葡萄酒芳香物质.xlsx")

    # 去除空值多的特征
    summary = data.isnull().sum()
    delete_list = summary >= 10  # 如果该列缺失值超过10个就舍弃这一列
    summary[delete_list]  # 删除的列的columns和该列缺失值个数
    data_select = data.drop(data.columns[delete_list], axis=1)

    # 缺失值如果是数值类型填充平均数，如果是离散类型填充众数
    func = lambda x: x.fillna(x.mean()) if x.values.dtype != object else x.fillna(x.mode()[0])
    data_select = data_select.apply(func, axis=0)

    return data_select

###############################################
import seaborn as sns
import matplotlib.pylab as plt

#plt图显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#读取数据
data_select = data_clean()
data_analysis = data_select

#相关性分析
corr_matrix = data_analysis.corr()
sns.heatmap(corr_matrix, vmax=1,square=True)


sub_matrix = corr_matrix["乙醛"]
sub_matrix.sort_values()[::-1][:10]   #与乙醛最正相关的前10个特征   [::-1]相当于[0:len():-1] 跨度为-1 相当于倒着数  [:]相当于[0:len():1]
sub_matrix.sort_values()[:10]   #与乙醛最负相关的前10个特征

