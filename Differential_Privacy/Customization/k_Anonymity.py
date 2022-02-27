''' K-匿名
数据的标识符，发布数据需要注意的地方
1、显式标识符（ID，能够唯一地确定一条用户记录）。
2、准标识符（QI，能够以较高的概率结合一定的外部信息确定一条用户记录）：单列并不能定位个人，但是多列信息可用来潜在的识别某个人。
3、敏感属性（需要保护的信息）。
4、非敏感属性（一般可以直接发布的信息）。

个人标识泄露。当数据使用人员通过任何方式确认数据表中某条数据属于某个人时，称为个人标识泄露。个人标识泄露最为严重，因为一旦发生个人标识泄露，数据使用人员就可以得到具体个人的敏感信息。

属性泄露，当数据使用人员根据其访问的数据表了解到某个人新的属性信息时，称为属性泄露。个人标识泄露肯定会导致属性泄露，但属性泄露也有可能单独发生。

成员关系泄露。当数据使用人员可以确认某个人的数据存在于数据表中时，称为成员关系泄露。成员关系泄露相对风险较小，个人标识泄露与属性泄露肯定意味着成员关系泄露，但成员关系泄露也有可能单独发生。

k-Anonymity 特点
1、k-Anonymity 是数据的一个属性，它确保每个个体都与至少一组k kk个体"融合"。
2、k-Anonymity 甚至在计算上也很昂贵：朴素算法是O（n^2），更快的算法占用相当大的空间。
3、k-Anonymity 可以通过泛化数据集来修改数据集来实现，这样特定值变得更加常见，组更容易形成。
'''

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
raw_data = {
  'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
  'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
  'age': [42, 52, 36, 24, 73],
  'preTestScore': [4, 24, 31, 2, 3],
  'postTestScore': [25, 94, 57, 62, 70]
}
df = pd.DataFrame(raw_data, columns=['age', 'preTestScore', 'postTestScore'])
print(df)

# 定义一个 k-匿名函数
def isK_Anonymized(df, k):
  for index, row in df.iterrows():
    query = ' & '.join([f'{col} == {row[col]}' for col in df.columns])
    rows = df.query(query)
    # print(rows)
    if len(rows) < k:
      return False
  return True

print(isK_Anonymized(df, 1))
print(isK_Anonymized(df, 2))
