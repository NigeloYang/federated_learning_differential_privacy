import matplotlib.pyplot as plt

'''
pycharm中plt.imshow()不显示图片
1、有的时候会遇到同样的环境和代码，在jupyter notebook里可以显示Matplotlib绘图
  但是在pycharm里无法显示Matplotlib绘图的情况。这是由于matplotlib.imshow()导致的，
  在绘图参数等语句和plt.imshow()之后加上plt.show()，即可正常显示Matplotlib绘图。

2、如果加 plt.show() 之后还是无法显示，可以试试首先导入pylab包
  然后在plt.imshow(img)后面添加一行代码 pylab.show()
'''

# 读取图片
img = plt.imread('image/cat.jpg')
plt.figure()

# 1、图片的基本操作
# plt.imshow(img)
# fig = plt.figure(2, figsize=(1.5, 1))
# small = img[:50, :40, :]
# plt.imshow(small)
# # 保存图片
# plt.imsave(r'.\image\cat_small.jpg', small)

# 2、图像色彩处理

# plt.subplot(2, 2, 1, title='Yellow Cat!')
# plt.imshow(img)
#
# plt.subplot(2, 2, 2, title='Pseudo color0 Cat!')
# img_r = img[:, :, 0]  # 取单通道-r通道,伪彩色
# plt.imshow(img_r)
#
# plt.subplot(2, 2, 3, title='Pseudo color1 Cat!')
# img_r1 = img[:, :, 1]  # 取单通道-g通道,伪彩色
# plt.imshow(img_r1)
#
# plt.subplot(2, 2, 4, title='Gray Cat!')
# img_r2 = img[:, :, 2]  # 取单通道-b通道,并指定灰度色
# plt.imshow(img_r2, plt.cm.gray)

# 3、加背景
# img_r = img[:, :, 0]
# plt.subplot(121, title='加背景火热色')
# plt.imshow(img_r, cmap="hot")
# plt.colorbar()
#
# plt.subplot(122, title='加背景冬天色')
# plt.imshow(img_r, cmap="winter")
# plt.colorbar()

plt.show()
