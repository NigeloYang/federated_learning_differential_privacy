''' 差分隐私的属性

1、顺序组合性质（Sequential composition）
2、平行组合（Parallel composition）
3、后处理（Post processing）
'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", family='MicroSoft YaHei')

# 顺序组合性质（Sequential composition）
# 如果 F1(x) 满足 ϵ1 / epsilon1 并且 F2(x) 满足 ϵ2 / epsilon2
# 则释放两个结果的机制 G(x) = (F1(x), F2(x)) 满足 ϵ1 + ϵ2
# 顺序组合是差分隐私的重要属性，因为它可以设计出多次查阅数据的算法。
epsilon1 = 1
epsilon2 = 1
epsilon3 = 2


# satisfies 1-differential differential_privacy
def F1():
    return np.random.laplace(loc=0, scale=1 / epsilon1)


# satisfies 1-differential differential_privacy
def F2():
    return np.random.laplace(loc=0, scale=1 / epsilon2)


# satisfies 2-differential differential_privacy
def F3():
    return np.random.laplace(loc=0, scale=1 / epsilon3)


# satisfies 2-differential differential_privacy, by sequential composition
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

# 并行组合性质（Parallel composition）
# 如果 F(x) 满足 ϵ / epsilon. 将数据集切分为 k 个互不相交的子数据块
# 则发布所有结果 F(x1),..., F(xk) 满足 ϵ

# 注意到，并行组合性给出的隐私消耗量比串行组合性要好得多。如果我们运行k次，串行组合性告诉我们这个过程满足 kϵ-差分隐私性，
# 而并行组合性告诉我们总隐私消耗量仅为 ϵ

# 获取原始数据集
adult = pd.read_csv('./data/adult_with_pii.csv')
print('原始数据集', adult.shape)
print(adult.head(2))

result = adult['Education'].value_counts().to_frame().head()
print('no laplace:  \n', result)

epsilon = 0.2

# This analysis has a total differential_privacy cost of epsilon = 1, even though we release many results!
lap = lambda x: x + np.random.laplace(loc=0, scale=1 / epsilon)
result_lap = adult['Education'].value_counts().apply(lap).to_frame().head()
print('add laplace: \n', result_lap)

# 列联表（Contingency Table）也被称为交叉列表（Cross Tabulation），有时也被简称为交叉表（Crosstab）。可以把列联表看成一个高维直方图。
print('Contingency Table \n ')
edu_sex = pd.crosstab(adult['Education'], adult['Sex']).head(5)
print('\n sdu_sex counts, no laplace: \n', edu_sex)

edu_sex_noise = pd.crosstab(adult['Education'], adult['Sex'])
f = lambda x: x + np.random.laplace(loc=0, scale=1 / epsilon)
res_edu_lap = edu_sex_noise.applymap(f).head(5)

print('\n sdu_sex counts, add laplace noise: \n', res_edu_lap)
