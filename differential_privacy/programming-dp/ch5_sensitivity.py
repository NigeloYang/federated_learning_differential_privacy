# -*- coding: utf-8 -*-
# @Time    : 2024/4/28

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", family='MicroSoft YaHei')

# 敏感度
'''我们如何确定特定函数的敏感度呢？对于实数域上的一些简单函数，答案是显而易见的。

f(x) = x 的全局敏感度是1，因为 x 变化1，f(x)变化为1
f(x) = x + x 的全局敏感度是2，因为 x 变化1，f(x) 变化为2
f(x) = 5x 的全局敏感度是5，因为 x 变化1，f(x) 变化为5
f(x) = x*x 的全局敏感度是无界的，因为 f(x) 的变化取决于 x 的值

对于将数据集映射到实数的函数，我们都采用类似的分析方法。我们下面将考虑3个常见的数据库聚合问询函数:计数问询、求和问询、均值问询。
'''

# 获取原始数据集
adult = pd.read_csv('./data/adult_with_pii.csv')
print('原始数据集', adult.shape)
print(adult.head(2))

# 计数问询
print('数据集中有多少人: ', adult.shape[0])
print('受教育年数超过10年的有多少人: ', adult[adult['Education-Num'] > 10].shape[0])
print('受教育年数小于或等于10年的有多少人: ', adult[adult['Education-Num'] <= 10].shape[0])
print('名字叫Joe Near的有多少人:', adult[adult['Name'] == 'Joe Near'].shape[0])

# 求和问询
# 一般来说，当待求和的属性值不存在上界和下界时，我们称求和问询具有无界敏感度。
# 当存在上下界时，求和问询的敏感度等于上下界的差
# 在下一节，我们将介绍裁剪（Clipping）技术。此技术用于在边界不存在时强制划定边界，以便将无界敏感度的求和问询转化为有界敏感度的问询。
print('受教育年数超过10年的人，其年龄总和是多少:', adult[adult['Education-Num'] > 10]['Age'].sum())

# 均值问询
print('受教育年数超过10年的人，其平均年龄是多少:', adult[adult['Education-Num'] > 10]['Age'].mean())

# 应用差分隐私回复均值问询的最简单方法是，将均值问询拆分为两个问询：求和问询除以计数问询。对上述例子，我们有
print(adult[adult['Education-Num'] > 10]['Age'].sum() / adult[adult['Education-Num'] > 10]['Age'].shape[0])


## 裁剪
# 差分隐私的拉普拉斯机制的无法直接应用于无界敏感度问询。幸运的是，我们通常可以利用裁剪（Clip）技术将此类问询转换为等价的有界敏感度问询。
# 裁剪技术的基本思想是，强制设置属性值的上界和下界。
print('无裁剪：', adult['Age'].sum())
print('裁剪上界 125：',adult['Age'].clip(lower=0, upper=125).sum())
print('裁剪上界 60：',adult['Age'].clip(lower=0, upper=60).sum())

plt.hist(adult['Age'])
plt.xlabel('年龄')
plt.ylabel('数据量')
plt.show()


# 如果我们通过查看数据来选择裁剪边界，那么边界本身也可能会泄露数据的一些相关信息。
# 一般先将敏感度下界设置为0，随后逐渐增加上界，直至问询输出不再变化（也就是说，即使进一步提高上界，问询的数据也不会再因裁剪而发生任何变化）
# 例如，让我们尝试计算裁剪边界从0到100的年龄总和，并对每次问询使用拉普拉斯机制，保证此过程满足差分隐私
def laplace_mech(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity/epsilon)

epsilon_i = .01
plt.plot([laplace_mech(adult['Age'].clip(lower=0, upper=i).sum(), i, epsilon_i) for i in range(100)])
plt.xlabel('年龄的裁剪边界')
plt.ylabel('总求和值')
plt.show()

# 我们可以用相同的方法估计任意数值型属性列的边界，但估计前我们最好能提前知道数据的大致取值范围。
# 例如，如果将年收入的边界值裁剪为0到100，裁剪边界的估计效果就不是很好，我们甚至无法找到合理的上界。
# 当数据的取值范围未知时，一种很好的改进方法就是根据依对数取值范围估计上界。

xs = [2**i for i in range(15)]
plt.plot(xs, [laplace_mech(adult['Age'].clip(lower=0, upper=i).sum(), i, epsilon_i) for i in xs])
plt.xscale('log')
plt.xlabel('年龄的裁剪边界')
plt.ylabel('总求和值')
plt.show()
