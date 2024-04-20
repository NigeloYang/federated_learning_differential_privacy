from sklearn.svm import LinearSVR
from sklearn.datasets import load_boston
from pandas import DataFrame

boston = load_boston()
data = DataFrame(boston.data, columns=boston.feature_names)
print('显示波士顿房价的前几行')
print(data.head(5))

data.insert(0, 'target', boston.target)
print('将价格添加到数据集中')
print(data.head(5))

data_mean = data.mean()
print('打印数据集平均值的前5行')
print(data_mean)

data_std = data.std()
print('打印数据集标准值的前5行')
print(data_std)

# 数据标准化
data_train = (data - data_mean) / data_std
print('打印数据集标准化后的前5行')
print(data_train.head(5))

x_train = data_train[boston.feature_names].values
y_train = data_train['target'].values
linearsvr = LinearSVR(0.1)
linearsvr.fit(x_train, y_train)

x = ((data[boston.feature_names] - data_mean[boston.feature_names]) / data_std[boston.feature_names]).values

data[u'y_pred'] = linearsvr.predict(x) * data_std['target'] + data_mean['target']
print(data[['target', 'y_pred']])
