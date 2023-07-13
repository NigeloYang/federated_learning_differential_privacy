# -*- coding: utf-8 -*-
# @Time    : 2023/6/21

'''剑指 Offer 17. 打印从 1 到最大的 n 位数
输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999
'''
from typing import List


class Solution:
    def printNumbers(self, n: int) -> List[int]:
        # 默认结果不越界
        # temp = 1
        # for i in range(n):
        #     temp *= 10
        # return [i for i in range(1, temp)]
        
        # 分治 作者：Krahets 带左边界带0
        # def dfs(x):
        #     if x == n:  # 终止条件：已固定完所有位
        #         res.append(''.join(num))  # 拼接 num 并添加至 res 尾部
        #         return
        #     for i in range(10):  # 遍历 0 - 9
        #         num[x] = str(i)  # 固定第 x 位为 i
        #         dfs(x + 1)  # 开启固定第 x + 1 位
        #
        # num = ['0'] * n  # 起始数字定义为 n 个 0 组成的字符列表
        # res = []  # 数字字符串列表
        # dfs(0)  # 开启全排列递归
        # return ','.join(res)  # 拼接所有数字字符串，使用逗号隔开，并返回
        
        # 左边界不带 0
        def dfs(x):
            if x == n:
                s = ''.join(num[self.start:])
                # 字符串处理大数
                # if s != '0': res.append(s)
                # 整数处理大数
                if s != '0':
                    res.append(int(s))
                if n - self.start == self.nine:
                    self.start -= 1
                return
            for i in range(10):
                if i == 9:
                    self.nine += 1
                num[x] = str(i)
                dfs(x + 1)
            self.nine -= 1
        
        num, res = ['0'] * n, []
        self.nine = 0
        self.start = n - 1
        dfs(0)
        return res


if __name__ == "__main__":
    print(Solution().printNumbers(1))
