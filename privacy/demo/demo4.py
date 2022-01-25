''' 差分隐私的属性
1、顺序组成（Sequential composition）
如果 F1(x) 满足 ϵ1 / epsilon_1 并且 F2(x) 满足 ϵ2 / epsilon_2
则释放两个结果的机制 G(x) = (F1(x), F2(x)) 满足 ϵ1 + ϵ2
顺序组合是差分隐私的重要属性，因为它可以设计出多次查阅数据的算法。

2、平行组合（Parallel composition）


3、后处理（Post processing）


'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
epsilon1 = 1
epsilon2 = 1
epsilon3 = 2

# satisfies 1-differential privacy
def F1():
    return np.random.laplace(loc=0, scale=1/epsilon1)

# satisfies 1-differential privacy
def F2():
    return np.random.laplace(loc=0, scale=1/epsilon2)

# satisfies 2-differential privacy
def F3():
    return np.random.laplace(loc=0, scale=1/epsilon3)

# satisfies 2-differential privacy, by sequential composition
def F_combined():
    return (F1() + F2()) / 2

# plot F1
plt.hist([F1() for i in range(1000)], bins=50, label='F1');

# plot F2
plt.hist([F2() for i in range(1000)], bins=50, alpha=1, label='F2');

# plot f3
plt.hist([F3() for i in range(1000)], bins=50, alpha=1, label='F3');

plt.hist([F_combined() for i in range(1000)], bins=50, alpha=1, label='F_combine');

plt.legend()
plt.show()

#
names = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education_Num', 'Marital_Status', 'Occupation', 'Relationship',
         'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Hours_per_week', 'Country', 'income']

# 获取原始数据集
adult_data = pd.read_csv('../data/adult.csv', names=names)
result = adult_data['Education'].value_counts().to_frame().head(5)
print('no laplace: ', result)

epsilon = 0.2

# This analysis has a total privacy cost of epsilon = 1, even though we release many results!
lap = lambda x: x + np.random.laplace(loc=0, scale=1/epsilon)
result_lap = adult_data['Education'].value_counts().apply(lap).to_frame().head(5)
print('add laplace: ', result_lap)