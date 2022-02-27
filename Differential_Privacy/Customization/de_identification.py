'''
去标识化代码实现
去识别化是从数据集中删除标识信息（个人身份证，地址，姓名等）的过程。术语去标识化有时与术语匿名化(de-identification)和假名化（pseudonymization）同义
而且去标识化容易受到链接攻击和差分攻击

案例是基于 UCI 的人口普查数据集, 因为数据集本身就是去标识化的，所以尝试一下去标识化容易受到链接攻击和差分攻击是怎么回事
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education_Num', 'Marital_Status', 'Occupation', 'Relationship',
         'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Hours_per_week', 'Country', 'income']

# 获取原始数据集
adult_data = pd.read_csv('../data/adult.csv', names=names)
print('原始数据集', adult_data.shape)
print(adult_data.head())

# 创建一个用于发布的数据集
adult_pre = adult_data.copy().drop(columns=['Education', 'Education_Num'])
print('发布的数据集')
print(adult_pre.shape)
print(adult_pre.head())

# 创建一个攻击者知道数据信息
adult_know = adult_data[['Marital_Status', 'Sex']]
print('攻击者知道数据信息')
print(adult_know.shape)
print(adult_know.head())

'''复现链接攻击
想象一下，由于发布的数据集的受教育程度，受教育时常，均已被删除，但碰巧知道一些关于这个人的辅助信息。
比如知道这个人的'Marital Status', 'Occupation', 'Race', 'Sex' 也就是结合数据集 adult_know & adult_pre 从而获取数据集。
为了执行简单的链接攻击，尝试攻击的数据集和所知道的辅助数据之间的重叠列。
'''
# Marital_Status_row = adult_know[adult_know['Marital_Status'] == 'Married-civ-spouse' ]
Marital_Status_row = adult_know.loc[adult_know['Marital_Status'] == 'Never-married']
# Marital_Status_row = adult_know.iloc[:, :][adult_know.Marital_Status == 'Married-civ-spouse']
print(Marital_Status_row)
get_data = pd.merge(Marital_Status_row, adult_pre, how='left', on=['Sex'])
print('打印获取到的数据集')
print(get_data)


adult_know['Marital_Status'].value_counts().hist()
print(adult_know['Marital_Status'].value_counts())
