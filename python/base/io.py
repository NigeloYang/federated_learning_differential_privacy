#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/22 23:18
# @Author : NigeloYang


# 检查 能否正常关闭读取的文件
# try:
#     f = open('./demo1.txt')
#     print(f.read())
# finally:
#     f.close()

# 升级版能否正常关闭文件
# with open('./demo1.txt', 'r') as f:
#     print('---1  f.read()---')
#     print(f.read())
#
# with open('./demo1.txt', 'r') as f:
#     print('---2 f.readline()---')
#     print(f.readline())
#
# with open('./demo1.txt', 'r') as f:
#     print('---3 f.readlines()---')
#     print(f.readlines())
#     for line in f.readlines():
#         line = line.strip()
#         print(line)
#
# print('---4 for .. in f.readlines()---')
# fo = open('./demo1.txt', 'r+')
# for line in fo.readlines():
#     line = line.strip()
#     print(line)
# fo.close()


# 探究文件操作
# import os
# import pickle
#
# print(os.name)
# print(os.environ)
# print(os.path.abspath('.'))
# print(os.path.join('\\ab\c', 'd'))
# print(os.path.split('\\b\c\\file.txt'))
# print(os.path.splitext('\\b\c\\file.txt'))
# for x in os.listdir(os.path.abspath('.')):
#     print(x)
#
# d = dict(name='Bob', age='20', score=89)
#
# f = open('./dump.txt', 'wb')
# pickle.dump(d, f)
# f.close()
#
# f = open('./dump.txt', 'rb')
# print(pickle.load(f))
# f.close()

from turtle import *

# 设置色彩模式是RGB:
colormode(255)

lt(90)

lv = 14
l = 120
s = 45

width(lv)

# 初始化RGB颜色:
r = 0
g = 0
b = 0
pencolor(r, g, b)

penup()
bk(l)
pendown()
fd(l)

def draw_tree(l, level):
    global r, g, b
    # save the current pen width
    w = width()

    # narrow the pen width
    width(w * 3.0 / 4.0)
    # set color:
    r = r + 1
    g = g + 2
    b = b + 3
    pencolor(r % 200, g % 200, b % 200)

    l = 3.0 / 4.0 * l

    lt(s)
    fd(l)

    if level < lv:
        draw_tree(l, level + 1)
    bk(l)
    rt(2 * s)
    fd(l)

    if level < lv:
        draw_tree(l, level + 1)
    bk(l)
    lt(s)

    # restore the previous pen width
    width(w)

speed("fastest")

draw_tree(l, 4)

done()
