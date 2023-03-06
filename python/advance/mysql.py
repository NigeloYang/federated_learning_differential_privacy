#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/25 23:10
# @Author : RicahrdYang
import pymysql

'''示例1'''
# # 连接数据库
# conn = pymysql.connect(
#     host='localhost',
#     user='root',
#     password='root',
# )
#
# # 生成游标对象
# cursor = conn.cursor()
#
# # 获取所有数据库
# cursor.execute("show databases")
# for i in cursor:
#     print(i)
#
# # 关闭游标和连接
# cursor.close()
# conn.close()


'''示例2'''
# 连接数据库
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='root',
    database='test'
)

# 生成游标对象
cursor = conn.cursor()

# 执行SQL语句
sql = "select * from student"
cursor.execute(sql)

# 获取结果集
result = cursor.fetchall()
print(result)

# 关闭游标和连接
cursor.close()
conn.close()
