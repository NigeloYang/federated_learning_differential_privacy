#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/25 22:52
# @Author : RicahrdYang

import re

# 实例1
# print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
# print(re.match('com', 'www.runoob.com'))         # 不在起始位置匹配

# 实例2
# line = "Cats are smarter than dogs"
# # .* 表示任意匹配除换行符（\n、\r）之外的任何单个或多个字符
# # (.*?) 表示"非贪婪"模式，只保存第一个匹配到的子串
# matchObj = re.match(r'(.*) are (.*?) (.*?) .*', line, re.M | re.I)
#
# if matchObj:
#     print("matchObj.group() : ", matchObj.group())
#     print("matchObj.group(1) : ", matchObj.group(1))
#     print("matchObj.group(2) : ", matchObj.group(2))
#     print("matchObj.group(2) : ", matchObj.group(3))
#     print("matchObj.group() : ", matchObj.groups())
# else:
#     print("No match!!")

# 实例3