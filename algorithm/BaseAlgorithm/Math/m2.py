#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/20 23:05
# @Author : RichardYang

'''计数质数
给定整数 n ，返回 所有小于非负整数 n 的质数的数量 。

示例 1：
输入：n = 10
输出：4
解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。

示例 2：
输入：n = 0
输出：0

示例 3：
输入：n = 1
输出：0
'''
class Solution:
    def countPrimes(self, n: int) -> int:
        isNumPrimes = [True] * n  # 将所有数，展开所有数 标记质数真
        count = 0  # 质数计数器 因为1不是质数 所以 0
        # 遍历2，n 数，判断是否是质数，从2开始对应-质数3 [1,2,3]  1不算质数
        for i in range(2, n):
            if isNumPrimes[i]:
                count += 1
                # 使用埃拉托斯特尼 筛选法进行过滤 将合数去除
                for j in range(i * i, n, i):  # 遍历 i*i  2倍i值 开始，结束n, 步数i (倍数递增)
                    isNumPrimes[j] = False  # 把合数置为 False
        return count

if __name__ == '__main__':
    print(Solution().countPrimes(5))
    print(Solution().countPrimes(10))
    print(Solution().countPrimes(100))
