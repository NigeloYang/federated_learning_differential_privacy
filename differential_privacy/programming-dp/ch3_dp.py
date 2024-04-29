'''差分隐私

拉普拉斯机制
laplace 数学方法：p(x) = 1/2b * exp(−|x| / b)

Laplace机制就是给查询结果添加服从Laplace分布的噪声，即：M(D) = f(D) + Lap(ϵ/s)

s 是 f 的敏感度，Lap(s)表示从中心为 0 且比例为 s 的拉普拉斯分布中采样
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font", family='MicroSoft YaHei')

# 获取原始数据集
adult = pd.read_csv('./data/adult_with_pii.csv')
print('原始数据集', adult.shape)
print(adult.head(2))


# 查询年龄在40 - 50 之间的
print('no laplace: ', adult[(adult['Age'] >= 40) & (adult['Age'] <= 50)].shape[0])

# 加入Laplace 噪声
epsilon = 0.1
sensitivity = 1

get_data = adult[(adult['Age'] >= 40) & (adult['Age'] <= 50)].shape[0]
lap_noise = np.random.laplace(loc=0, scale=sensitivity/epsilon)

print('add laplace epsilon = 0.1: ', get_data + lap_noise)


# 加入Laplace 噪声
epsilon = 0.2
sensitivity = 1

get_data = adult[(adult['Age'] >= 40) & (adult['Age'] <= 50)].shape[0]
lap_noise = np.random.laplace(loc=0, scale=sensitivity/epsilon)

print('add laplace epsilon = 0.2: ', get_data + lap_noise)


# 尝试攻击

# 如何知道拉普拉斯机制是否已经增加了足够的噪声，可以阻止攻击者对数据集中的个体实施重标识攻击？我们可以先尝试自己来实施攻击！
# 我们构造一个恶意的计数问询，专门用于确定凯莉·特鲁斯洛夫的收入是否大于$50k。

karries_row = adult[adult['Name'] == 'Karrie Trusslove']
res = karries_row[karries_row['Target'] == '<=50K'].shape[0]
print('结果为 1 ，则扰动太小,泄露隐私', res)


# 为查询结果添加差分噪声
sensitivity = 1
epsilon = 0.1

karries_row = adult[adult['Name'] == 'Karrie Trusslove']
res = karries_row[karries_row['Target'] == '<=50K'].shape[0] + np.random.laplace(loc=0, scale=sensitivity/epsilon)
print('结果为 1 ，则扰动太小,泄露隐私，不为1 则正常扰动', res)