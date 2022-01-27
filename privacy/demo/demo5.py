''' 高斯噪音
高斯机制是拉普拉斯机制的替代方案，不同在于加的是高斯噪声而不是拉普拉斯噪声。
高斯机制不满足纯 ϵ-差分隐私，但满足 (epsilon, delta) (ϵ,δ)-差分隐私。

公式：Ｆ(x) = f(x) + N(σ**2)
σ**2 = (2 * s**2 * log(1.25/δ)) / ϵ**2

s 是 f 的敏感度，而 N(σ**2) 表示从中心为 0 且方差 σ**2 (sigma**2) 的高斯（正态）分布抽样

如何计算 敏感度 一般使用 L1 and  L2 norm
'''


import matplotlib.pyplot as plt
from math import exp
plt.style.use('seaborn-whitegrid')
import pandas as pd
import numpy as np

# delta = 10e-5
# epsilon = 1
# sensitivity = 1
#
# laplace = [np.random.laplace(loc=0, scale=1/epsilon) for x in range(100000)]
#
# sigma = np.sqrt(2 * sensitivity**2 * np.log(1.25 / delta) / epsilon**2 )
# gauss = [np.random.normal(loc=0, scale=sigma) for x in range(100000)]
#
# plt.hist(laplace, bins=50, label='Laplace')
# plt.hist(gauss, bins=50, alpha=.8, label='Gaussian')
# plt.legend()
#
# plt.show()

print(exp(25)/(exp(25)+exp(15)+exp(10)))