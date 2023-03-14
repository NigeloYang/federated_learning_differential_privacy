#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/14 22:50
# @FileName : network_serve.py
# @Author : RicahrdYang

import socket
import sys

def main():
    # 创建socket对象
    serversocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    # 获取本地主机名
    host = socket.gethostname()
    port = 9999

    # 绑定端口号
    serversocket.bind((host,port))

    # 设置最大的连接数，超过后排队
    serversocket.listen(5)

    while True:
        # 建立客户端连接
        clientsocket,addr = serversocket.accept()
        print("connect address: %s"%str())

        msg = "测试成功！" + "\r\n"
        clientsocket.send(msg.encode('utf-8'))
        clientsocket.close()

if __name__ == "__main__":
    main()