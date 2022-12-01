#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/27 22:58
# @Author : NigeloYang

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# adult_s = pd.read_csv('../data/adult.csv')
# print(adult_s.shape)
# print(adult_s.head())

adult = pd.read_csv('../data/adult_with_pii.csv')
print(adult.shape)
print(adult.head())

domain = adult['Occupation'].dropna().unique()
print(domain)


# 本地差分隐私-随机响应
def rand_resp_sales(response):
    truthful_response = response == 'Sales'

    # 第一次随机抛掷硬币
    if np.random.randint(0, 2) == 0:
        # 如实回答问题
        return truthful_response
    else:
        # 第二次抛掷硬币随机应答
        return np.random.randint(0, 2) == 0


# 测试200个销售的回答
print(pd.Series([rand_resp_sales('Sales') for i in range(200)]).value_counts(),'\n')

# 测试人口数据集中的销售人员数量
response = [rand_resp_sales(r) for r in adult['Occupation']]
print(f'dp sales: \n {pd.Series(response).value_counts()} \n')

# 真实销售人员数量
origin_yeses = len(adult[adult["Occupation"] == "Sales"])
print(f'True sales: {origin_yeses} \n')

# 对进行本地差分隐私的数据执行无偏差估计
# 估计有1/4回答“是”的数据是来自随机抛掷硬币结果得到
fake_yeses = len(response)/4
num_yeses = np.sum([1 if r else 0 for r in response])
true_yeses = num_yeses -fake_yeses

# 用true_yesses估计"真实"组中回答"是"的人数,我们把人数翻倍，估计出回复为"是"的总人数
rr_result = true_yeses *2
print(f'debias dp sales: {rr_result} \n')


print(f'ldp percentage error: {(origin_yeses - rr_result)/origin_yeses * 100:.3}%')
print(f'dp percentage error: {(origin_yeses - laplace_mech(origin_yeses,1,1) )/origin_yeses * 100:.3}%')
