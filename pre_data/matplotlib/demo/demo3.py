from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib

# 避免中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(40, 15))

# 1、初步尝试绘制3d图
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# plt.subplot(121,projection='3d')


# 2、以别名方式导入三维坐标模块
# from matplotlib import pyplot as plt
# import mpl_toolkits.mplot3d as p3d  # 以别名方式导入三维坐标模块
#
# fig = plt.figure()
# ax2 = p3d.Axes3D(fig)  # 在当前figure上建立三维坐标
# ax2.set_xlim(0, 6)  # X轴，横向向右方向
# ax2.set_ylim(7, 0)  # Y轴，左向与X、Z轴互为垂直
# ax2.set_zlim(0, 8)  # 竖向为Z轴


# 2、绘制点
dot1 = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 2, 4]]  # 五个(x,y,z)点
ax1 = plt.subplot(2, 5, 1, projection='3d')
ax1.set_xlim(0, 5)
ax1.set_ylim(5, 0)
ax1.set_zlim(0, 5)
color1 = ['r', 'g', 'b', 'k', 'm']
marker1 = ['o', 'v', '1', 's', 'H']
i = 0
for x in dot1:
  ax1.scatter(x[0], x[1], x[2], c=color1[i], marker=marker1[i], linewidths=4)  # 用散点函数画点
  i += 1

# 3, 绘制线
ax3 = plt.subplot(2, 5, 2, projection='3d')
ax3.set_xlim(0, 20)  # X轴，横向向右方向
ax3.set_ylim(20, 0)  # Y轴，左向与X、Z轴互为垂直
ax3.set_zlim(0, 20)  # 竖向为Z轴
z = np.linspace(0, 4 * np.pi, 500)
x = 10 * np.sin(z)
y = 10 * np.cos(z)
ax3.plot3D(x, y, z, 'red')

z1 = np.linspace(0, 4 * np.pi, 500)
x1 = 5 * np.sin(z)
y1 = 5 * np.cos(z)
ax3.plot3D(x1, y1, z1, 'g--')

ax3.plot3D([0, 18, 0], [5, 18, 10], [0, 0, 0], 'om-')

# 4.1、绘制面
ax4 = plt.subplot(2, 5, 3, projection='3d')


def Z(X, Y):
  return X * 0.2 + Y * 0.3 + 20


ax4.set_xlim3d(0, 50)
ax4.set_ylim3d(0, 50)
ax4.set_zlim3d(0, 50)
x = np.arange(1, 50, 1)
y = np.arange(1, 50, 1)
X, Y = np.meshgrid(x, y)
# s = ax4.plot_surface(X, Y, Z(X, Y), rstride=10, cstride=10, cmap=cm.jet, linewidth=1, antialiased=True)
s = ax4.plot_surface(X, Y, Z(X, Y), rstride=1, cstride=1, cmap=cm.jet, linewidth=1, antialiased=False)
plt.colorbar(s, shrink=1, aspect=5)

# 4.2、绘制面
ax5 = plt.subplot(254, projection='3d')
d = 0.05


def z(x, y):
  res1 = np.exp(-x ** 2 - y ** 2)
  res2 = np.exp(-(x - 1) ** 2 - (y - 1) ** 2)
  return (res2 - res1) * 2


x1 = np.arange(-4.0, 4.0, d)
y1 = np.arange(-3.0, 3.0, d)
X1, Y1 = np.meshgrid(x1, y1)
s = ax5.plot_surface(X1, Y1, z(X1, Y1), rstride=1, cstride=1, cmap=cm.jet, linewidth=1, antialiased=False)
plt.colorbar(s, shrink=1, aspect=5)

# 5、绘制光源 见jupyter notebook demo1
# 绘制标签
ax6 = plt.subplot(255, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
ax6.plot_surface(x, y, z, color='m', cmap=plt.cm.winter, lightsource=(180, 35, 0, 0.2, 0.1, 0), alpha=0.7)
ax6.set_xlabel('x axis', fontsize=15)
ax6.set_ylabel('y axis', fontsize=15)
ax6.set_zlabel('z axis', fontsize=15)
ax6.text(0, 0, 0, 'ball', color='r', fontsize=17)
z1 = np.linspace(0, np.pi * 4, 500)
x1 = 10 * np.sin(z1)
y1 = 10 * np.cos(z1)
ax6.plot3D(x1, y1, z1, 'black', label='show me')
ax6.legend()

# 6、


plt.show()
