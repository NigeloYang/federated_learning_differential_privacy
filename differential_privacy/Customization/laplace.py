'''拉普拉斯机制

laplace 数学方法：p(x) = 1/2b * exp(−|x| / b)

Laplace机制就是给查询结果添加服从Laplace分布的噪声，即：M(D) = f(D) + Lap(ϵ/s)

s 是 f 的敏感度，Lap(s)表示从中心为 0 且比例为 s 的拉普拉斯分布中采样
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

names = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education_Num', 'Marital_Status', 'Occupation', 'Relationship',
         'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Hours_per_week', 'Country', 'income']

# 获取原始数据集
adult_data = pd.read_csv('../data/adult.csv', names=names)
print('原始数据集', adult_data.shape)
print(adult_data.head())

# 查询年龄在40 - 50 之间的
print('no laplace: ', adult_data[(adult_data['Age'] >= 40) & (adult_data['Age'] <= 50)].shape[0])

# 加入Laplace 噪声
epsilon = 0.1
sensitivity = 1

get_data = adult_data[(adult_data['Age'] >= 40) & (adult_data['Age'] <= 50)].shape[0]
lap_noise = np.random.laplace(loc=0, scale=sensitivity/epsilon)

print('add laplace epsilon = 0.1: ', get_data + lap_noise)


# 加入Laplace 噪声
epsilon = 0.2
sensitivity = 1

get_data = adult_data[(adult_data['Age'] >= 40) & (adult_data['Age'] <= 50)].shape[0]
lap_noise = np.random.laplace(loc=0, scale=sensitivity/epsilon)

print('add laplace epsilon = 0.2: ', get_data + lap_noise)

