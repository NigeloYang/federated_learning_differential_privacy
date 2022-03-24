import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

# 避免中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 移动刻度线
# x = np.linspace(-np.pi, np.pi, 100)
# c, s, t = np.cos(x), np.sin(x), np.arctan(x)
# plt.plot(x, c, r'r')
# plt.plot(x, s, r'g')
# plt.plot(x, t, r'b')
# plt.plot(x, np.arcsin(x), r'o')
# plt.plot(x, np.arccos(x), r'p')
# 获取当前 axes 实例
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# #  使用spines函数 对图像的数据抽进行了一个变换
# # 把 X 轴设置成图像的 bottom 边
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# # 把 Y 轴设置成图像的 left 边
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# plt.show()

'''
  绘制图形
'''
plt.figure(figsize=(40, 15))
# 1、绘制矩形
ax1 = plt.subplot(261, title='绘制矩形')
demo1 = plt.Rectangle((0.2, 0.2), 0.2, 0.2, color='r', alpha=0.8)
demo21 = plt.Rectangle((0.5, 0.5), 0.2, 0.4, color='orange', alpha=0.8)
demo2 = plt.Rectangle((0.5, 0.5), 0.2, 0.4, color='g', alpha=0.8, angle=60)
demo3 = plt.Rectangle((0.5, 0.2), 0.4, 0.2, color='b', alpha=0.8, linestyle='--')
ax1.add_patch(demo1)
ax1.add_patch(demo21)
ax1.add_patch(demo2)
ax1.add_patch(demo3)

# 2、绘制圆形 和 椭圆形
ax2 = plt.subplot(262, title='绘制圆形 和 椭圆形')
C1 = Circle(xy=(0.2, 0.2), radius=.2, alpha=0.5)
ax2.add_patch(C1)
E1 = Ellipse(xy=(0.6, 0.6), width=0.5, height=0.2, angle=30.0, facecolor='yellow', alpha=0.9)
ax2.add_patch(E1)

# 3、绘制多边形
ax3 = plt.subplot(263, title='绘制多边形')
p1 = plt.Polygon([[0.15, 0.15], [0.15, 0.7], [0.4, 0.15]], color='k', alpha=0.5)
p2 = plt.Polygon([[0.45, 0.15], [0.2, 0.7], [0.55, 0.7], [0.8, 0.15]], color='g', alpha=0.9)
p3 = plt.Polygon([[0.69, 0.45], [0.58, 0.7], [0.9, 0.7], [0.9, 0.45]], color='b', alpha=0.9)

ax3.add_patch(p1)
ax3.add_patch(p2)
ax3.add_patch(p3)

# 4、绘制条形图
plt.subplot(264, title='条形图班级人数统计')
c = ['四年级', '五年级', '六年级']
x = np.arange(len(c)) * 0.8
girl = [19, 19, 22]
boy = [20, 18, 21]
b1 = plt.bar(x, height=girl, width=0.1, alpha=0.8, color='red', label='女生')
b2 = plt.bar([x1 + 0.1 for x1 in x], height=boy, width=0.1, alpha=0.8, color='green', label='男生')
plt.legend()
plt.ylim(0, 40)
plt.ylabel('人数')
plt.xticks([index + 0.1 for index in x], c)
plt.xlabel('班级')
for r1 in b1:
  height = r1.get_height()
  plt.text(r1.get_x() + r1.get_width() / 2, height + 1, str(height), ha='center', va='bottom')
for r2 in b2:
  height = r2.get_height()
  plt.text(r2.get_x() + r2.get_width() / 2, height + 1, str(height), ha="center", va="bottom")

# 5、绘制直方图
plt.subplot(265, title='绘制直方图')
d1 = np.random.randn(1000)
plt.hist(d1, bins=40, facecolor='blue', edgecolor='black', alpha=0.9)
plt.xlabel('概率分布区间')
plt.xlabel('频数/频率')
plt.title('频数/频率分布直方图')

# 6、绘制饼状图
ax6 = plt.subplot(266, title='班级春季生病原因比较')
label = ('感冒', '肠胃不适', '过敏', '其它疾病')
color = ('red', 'orange', 'yellow', 'green')
size = [48, 21, 18, 13]
explode = (0.1, 0, 0, 0)
ax6.pie(size, colors=color, explode=explode, labels=label, shadow=True, autopct='%1.1f%%')
ax6.axis('equal')
ax6.legend()

# 7、绘制散点图
ax7 = plt.subplot(267, title='绘制散点图')
n = 1000
# 产生正态分布的随机数1000个,在xy数轴显示
x = np.random.randn(n)
x = np.random.randn(n)
y = np.random.randn(n)
color = ['r', 'y', 'k', 'g', 'm'] * int(n / 5)
ax7.scatter(x, y, c=color, marker='o', alpha=0.8)

# 8、绘制极坐标图
plt.subplot(268, projection='polar', title='绘制极坐标')
t = np.arange(0, 2 * np.pi, 0.02)
plt.plot(t, t / 6, '--', lw=2)

# 9、绘制极等高图 plt.contour, 绘制等高轮廓线plt.contourf
plt.subplot(269, title='绘制等高图')


def f(x, y):
  return x ** 2 + y ** 2 - 1


n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='jet')
contour = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=0.25)
plt.clabel(contour, inline=True, fontsize=10)

# 10、图形填充 多边形plt.fill(), 图形区间填充， plt.fill_between()
plt.subplot(2, 6, 10, title='填充图形')
x = np.linspace(-np.pi, np.pi, 100)
y1 = np.sin(x)
y2 = np.sin(2 * x)
plt.fill(x, y1, facecolor='r', alpha=.7)
plt.fill(x, y2, facecolor='g', alpha=.7)

# 11、
plt.subplot(2, 6, 11, title='填充图形')
x = [1, 1, 2, 2, 3, 3, 4, 4]
y = [1, 2, 2, 1, 1, 2, 2, 1]
y1 = [1, 1, 1, 1, 1, 1, 1, 1]
plt.plot(x, y)
plt.plot(x, y1)
cmap = plt.cm.get_cmap("winter")  # 设置冬天的颜色
plt.fill_between(x, y, y1, alpha=0.7, hatch='/', cmap=cmap)

# 12、
plt.subplot(2, 6, 12, title='填充图形')
origin = 'lower'

delta = 0.025
x = y = np.arange(-3.0, 3.01, delta)
X, Y = np.meshgrid(x, y)
Z1 = plt.mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = plt.mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = 10 * (Z1 - Z2)
nr, nc = Z.shape
# put NaNs in one corner:
Z[-nr//6:, -nc//6:] = np.nan
# contourf will convert these to masked
Z = np.ma.array(Z)
# mask another corner:
Z[:nr//6, :nc//6] = np.ma.masked
interior = np.sqrt((X**2) + (Y**2)) < 0.5
Z[interior] = np.ma.masked
label=[-1.5, -1, -0.5, 0, 0.5, 1]
CS = plt.contourf(X, Y, Z, #10,
                  label,
                  #alpha=0.5,
                  #cmap=plt.cm.bone,
                  colors=('r', 'g', 'b'),
                  origin=origin)

CS2 = plt.contour(CS, levels=CS.levels[::2],
                  colors='r',
                  origin=origin)
plt.title('Nonsense (3 masked regions)')
plt.xlabel('word length anomaly')
plt.ylabel('sentence length anomaly')
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('verbosity coefficient')
cbar.add_lines(CS2)
plt.figure()
plt.clabel(CS, inline=True, fontsize=12)
plt.clabel(CS2, inline=True, fontsize=12)

plt.show()
