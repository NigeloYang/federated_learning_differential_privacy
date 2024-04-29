''' 高斯噪音
高斯机制是拉普拉斯机制的替代方案，不同在于加的是高斯噪声而不是拉普拉斯噪声。
高斯机制不满足纯 ϵ-差分隐私，但满足(ϵ,δ)-差分隐私。

公式：Ｆ(x) = f(x) + N(σ**2)
σ**2 = (2 * s**2 * log(1.25/δ)) / ϵ**2

s 是 f 的敏感度，而 N(σ**2) 表示从中心为 0 且方差 σ**2 (sigma**2) 的高斯（正态）分布抽样

- 平行组合
假设我们有一组隐私机制 `M = { M1， … ， Mm }` 如果每个 Mi 对整个数据集的互不相交子集提供 `ϵ-DP` 保证，则 M 将提供 `max {ϵ1，…，ϵm}-DP`
这个性质说明了, 当有多个算法序列分别作用在一个数据集上多个不同子集上时, 最终的差分隐私预算等价于算法序列中所有算法预算的最大值


- 顺序组合
假设在数据集上依次执行一组隐私机制 `M = { M1 ， . . . ， Mm }`，并且每个 Mi 提供 `ϵ-DP` 保证，则 M 将提供 `(m * ϵ)-DP`
这个性质说明了, 当有一个算法序列同时作用在一个数据集上时, 最终的差分隐私预算等价于算 法序列中所有算法的预算的和


- 高级组合
高级组合定理通常用机制来表示，这些机制是 k-fold 自适应组合的实例。k-fold 自适应组合是一系列机制`m_1, ……, m_k`，使得：
每个机制 m_i 都可以根据所有先前机制的输出来选择`m_1, ……, m_{i-1}`（因此自适应)。
每个机制 m_i 的输入既是私有数据集，也是以前机制的所有输出（因此组成)。
如果 k-fold 自适应组合`m_1, ……, m_k`中的每个机制 m_i 都满足`ϵ-差分隐私`,然后对于任何`δ ≥ 0`，整个 k-fold 自适应组合
满足`(ϵ′, δ′)-差分隐私`，其中`ϵ′ = 2ϵ * sqrt(2 * k * log(1/δ′))`
'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

delta = 10e-5
epsilon = 1
sensitivity = 1

laplace = [np.random.laplace(loc=0, scale=1 / epsilon) for x in range(100000)]

sigma = np.sqrt(2 * sensitivity ** 2 * np.log(1.25 / delta) / epsilon ** 2)
gauss = [np.random.normal(loc=0, scale=sigma) for x in range(100000)]

plt.hist(laplace, bins=50, label='Laplace')
plt.hist(gauss, bins=50, alpha=.8, label='Gaussian')
plt.legend()
plt.show()

# 高级组合，顺序组合
epsilon = 1
delta = 10e-5


def adv_comp(k):
  return 2 * epsilon * np.sqrt(2 * k * np.log(1 / delta))


def seq_comp(k):
  return k*epsilon

# 循环 100 epoch
plt.plot([seq_comp(k) for k in range(100)], label='Sequential Composition')
plt.plot([adv_comp(k) for k in range(100)], label='Advanced Composition')
plt.legend()
plt.show()

# 循环 10000 epoch
plt.plot([seq_comp(k) for k in range(10000)], label='Sequential Composition')
plt.plot([adv_comp(k) for k in range(10000)], label='Advanced Composition')
plt.legend()
plt.show()