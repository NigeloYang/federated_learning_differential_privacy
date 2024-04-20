from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 避免中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig = plt.figure()

# 图一
ax1 = Axes3D(fig)
x = np.linspace(0, 15, 100)
y = np.sin(x)
z = np.cos(x)
ax1.plot(x, y, z)

# 图二
ax2 = Axes3D(fig)
colors = ['r', 'g', 'b']  # 定义颜色列表
year = [2016, 2017, 2018]  # 定义年份列表
for z, color in zip(year, colors):
  x = range(1, 13)
  y = 100000 * np.random.rand(12)
  ax2.bar(x, y, zs=z, zdir='y', color=color, alpha=0.8)
ax2.set_xlabel('月份')
ax2.set_ylabel('年份')
ax2.set_zlabel('销量')
ax2.set_yticks(year)  # y轴只显示年份数据

plt.show()
