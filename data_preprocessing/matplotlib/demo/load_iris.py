from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib

# 避免中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

iris = load_iris()
print(iris.data[0])
print(iris.data[0, 0])
print(iris.data[1])
print(iris.data[1, 0])
print(iris.data[iris.target == 0, 0])

iris_length = 0
iris_width = 1
colors = ['red', 'green', 'blue']
for label, color in zip(range(len(iris.target_names)), colors):
  plt.scatter(iris.data[iris.target == label, iris_length],
              iris.data[iris.target == label, iris_width],
              label=iris.target_names[label],
              c=color)
plt.xlabel('萼片长度')
plt.ylabel('萼片宽度')
plt.legend()
plt.show()
