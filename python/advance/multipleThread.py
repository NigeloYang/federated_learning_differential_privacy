#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/15 23:03
# @FileName : multipleThread.py
# @Author : RicahrdYang


import threading
import time


class Thread_create(threading.Thread):
    def __init__(self, threadID, name, delay):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.delay = delay
        self.exitFlag = 0

    def run(self):
        print("开始线程：" + self.name)
        self.print_time(self.name, self.delay, 5)
        print("退出线程：" + self.name)

    def print_time(self, threadName, delay, counter):
        while counter:
            if self.exitFlag:
                threadName.exit()
            time.sleep(delay)
            print("%s: %s" % (threadName, time.ctime(time.time())))
            counter -= 1


class thread_sync(threading.Thread):
    def __init__(self, threadid, name, delay):
        threading.Thread.__init__(self)
        self.threadid = threadid
        self.name = name
        self.delay = delay
        self.threadLock = threading.Lock()

    def run(self):
        print("开启线程： " + self.name)
        # 获取锁，用于线程同步
        self.threadLock.acquire()
        self.print_time(self.name, self.delay, 3)
        # 释放锁，开启下一个线程
        self.threadLock.release()

    def print_time(self, threadName, delay, counter):
        while counter:
            time.sleep(delay)
            print("%s: %s" % (threadName, time.ctime(time.time())))
            counter -= 1


if __name__ == "__main__":
    threads = []

    # 创建新线程
    thread1 = thread_sync(1, "Thread-1", 1)
    thread2 = thread_sync(2, "Thread-2", 2)

    # 开启新线程
    thread1.start()
    thread2.start()

    # 添加线程到线程列表
    threads.append(thread1)
    threads.append(thread2)

    # 等待所有线程完成
    for t in threads:
        t.join()
    print("退出主线程")
