#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/6 19:53
# @File    : d1.py
# @Author  : Richard Yang

'''剑指 Offer 10- I. 斐波那契数列
您已有成功提交记录，请确认是否跳过？

跳过即视为本节
已完成


跳过

重新做题
写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

 

示例 1：

输入：n = 2
输出：1
示例 2：

输入：n = 5
输出：5
'''


class Solution:
    def fib(self, n: int) -> int:
        if n == 0:
            return 0
        elif n == 1 or n == 2:
            return 1
        else:
            a, b = 1, 1
            for i in range(2, n):
                a, b = b, a + b
            return b % 1000000007


if __name__ == "__main__":
    print(Solution().fib(3))
    print(Solution().fib(5))
