'''去标识化代码实现
去识别化是从数据集中删除标识信息（个人身份证，地址，姓名等）的过程。术语去标识化有时与术语匿名化(de-identification)和假名化（pseudonymization）同义
而且去标识化容易受到链接攻击和差分攻击，本章主要介绍以下内容：

1.定义并理解下述概念：
    1.1去标识
    1.2.重标识
    1.3.标识信息 / 个人标识信息
    1.4.关联攻击
    1.5.聚合与聚合统计
    1.6.差分攻击
2.实施一次关联攻击
3.实施一次差分攻击
4.理解去标识技术的局限性
5.理解聚合统计的局限性

案例是基于 UCI 的人口普查数据集, 所以尝试一下去标识化容易受到的链接攻击和差分攻击
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

# 移除个人标识信息,创建一个用于发布的数据集
adult_data = adult.copy().drop(columns=['Name', 'SSN'])
print('去除标识，创建发布数据集', adult_data.shape)
print(adult_data.head(2))

# 保留一部分信息作为辅助信息，看作攻击者知道的背景信息知识，发动一次重标识攻击（背景知识攻击）
adult_pii = adult[['Name', 'SSN', 'DOB', 'Zip']]
print('创建辅助信息集', adult_data.shape)
print(adult_pii.head(2))

'''关联攻击（Linkage Attack）
发布数据中的姓名一列已经被移除了，但我们碰巧知道能帮助我们标识出这位朋友的一些辅助信息。我们的这位朋友叫凯莉·特鲁斯洛夫（Karrie Trusslove），
我们知道凯莉的出生日期和邮政编码。

我们尝试攻击的数据集与我们知道的一些辅助信息集之间包含一些重叠列，我们将应用这些重叠列来实施一次简单的关联攻击（Linkage Attack）。
'''
karries_row = adult_pii[adult_pii['Name'] == 'Karrie Trusslove']
print('查询到的数据', karries_row)

get_data = pd.merge(karries_row, adult_data, left_on=['DOB', 'Zip'], right_on=['DOB', 'Zip'])
print('辅助信息集和发布数据集结合获取到的数据')
print(get_data)

# 重标识难度

# 简单场景中只需要一行数据就可以构建出指定信息
get_karrieszip = pd.merge(karries_row, adult_data, left_on=['Zip'], right_on=['Zip'])
print('简单场景，已知一条地区编码信息，通过辅助信息集和发布数据集结合获取到的数据')
print(get_karrieszip)

get_karriesdop = pd.merge(karries_row, adult_data, left_on=['DOB'], right_on=['DOB'])
print('简单场景，已知出生日期，通过辅助信息集和发布数据集结合获取到的数据')
print(get_karriesdop)

'''在数据集中重标识出其他某个个体的难度有多大？

衡量此类攻击有效性的一个好方法是查看特定数据是否有较好的”筛选效果”：特定数据能否帮助我们更好地缩小目标个体所属行的范围。
举个例子，数据集中拥有相同出生日期的人数多吗？
'''
print('统计出生日期分布 \n', adult_pii['DOB'].value_counts())
adult_pii['DOB'].value_counts().hist()
plt.title("DOB 筛选个体", loc="center")
plt.xlabel('生日数量')
plt.ylabel('出现次数')
plt.show()

print('统计邮政编码分布 \n', adult_pii['Zip'].value_counts())
adult_pii['Zip'].value_counts().hist()
plt.title("ZIP 筛选个体", loc="center")
plt.xlabel('邮政编码数量')
plt.ylabel('出现次数')
plt.show()

attack = pd.merge(adult_pii, adult_data, left_on=['DOB'], right_on=['DOB'])
print('统计DOB分布,结合辅助信息 \n', attack['Name'].value_counts())
attack['Name'].value_counts().hist()
plt.title("结合辅助信息 DOB，筛选个体", loc="center")
plt.xlabel('生日数量')
plt.ylabel('出现次数')
plt.show()

attack = pd.merge(adult_pii, adult_data, left_on=['DOB', 'Zip'], right_on=['DOB', 'Zip'])
print('统计DOB,ZIP分布,结合辅助信息 \n', attack['Name'].value_counts())
attack['Name'].value_counts().hist()
plt.title("结合辅助信息 DOB,ZIP，筛选个体", loc="center")
plt.xlabel('生日数量')
plt.ylabel('出现次数')
plt.show()

# 组合使用多种信息会得到非常好的筛选效果，通过组合出生日期、性别、邮政编码，可以唯一重标识出87%的美国公民
print('结合DOB,ZIP，查看重标识结果 \n', attack['Name'].value_counts().head(5))

'''一种防止隐私信息泄露的方法是只发布聚合（Aggregate）数据。

一般认为，对数据进行聚合处理可以提升数据的隐私保护效果，因为很难识别出特定个体对聚合统计结果所带来的影响。
如果某个分组只包含一个个体呢？在这种情况下，举个统计结果将准确泄露此个体的年龄，无法提供任何隐私保护
'''
print('发布聚合年龄结果 \n', adult['Age'].mean())

print('发布聚合年龄结果，以区域为小组 \n', adult[['Zip', 'Age']].groupby('Zip').mean().head())


'''差分攻击

当对相同的数据发布多个聚合统计结果时，隐私泄露问题会变得棘手。
例如，考虑对数据集中某个大分组执行两次求和问询（第一个是对整个数据集进行问询，第二个是对除一条记录外的所有记录进行问询）
'''
print('整体发布聚合年龄结果 \n', adult['Age'].sum())
print('扣除一条信息再发布聚合年龄结果 \n', adult[adult['Name'] != 'Karrie Trusslove']['Age'].sum())
print('两次查询年龄差 \n',adult['Age'].sum() - adult[adult['Name'] != 'Karrie Trusslove']['Age'].sum())


''' 总结
关联攻击指的是组合使用辅助数据和去标识数据来重标识个体。

实施关联攻击最简单的方法是：将数据集中的两个数据表关联起来。

即使实施简单的关联攻击，攻击效果也非常显著：

只需要一个辅助数据点，就足以把攻击范围缩小到几条记录

缩小后的记录可以进一步显示出哪些额外的辅助数据会有助于进一步实施攻击

对于一个特定的数据集，两个数据点一般足以重标识出绝大多数个体

三个数据点（性别、邮政编码、出生日期）可以唯一重标识出87%的美国公民
'''