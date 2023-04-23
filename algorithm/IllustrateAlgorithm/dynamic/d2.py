#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/8 13:48
# @File    : d2.py
# @Author  : Richard Yang
'''剑指 Offer 10- II. 青蛙跳台阶问题
一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

示例 1：
输入：n = 2
输出：2

示例 2：
输入：n = 7
输出：21

示例 3：
输入：n = 0
输出：1

提示：
0 <= n <= 100
注意：本题与主站 70 题相同：https://leetcode-cn.com/problems/climbing-stairs/
'''


class Solution:
    def numWays(self, n: int) -> int:
        # 方案1
        # if n <= 1:
        #     return 1
        # else:
        #     num, a, b = 0, 1, 1
        #     for _ in range(2, n+1):
        #         num = a + b
        #         a = b
        #         b = num
        #     return int(num % 1000000007)
        
        # 方案2
        if n <= 1:
            return 1
        else:
            dp = [0] * (n + 1)
            dp[0], dp[1] = 1, 1
            for i in range(2, n + 1):
                dp[i] = dp[i - 1] + dp[i - 2]
            return int(dp[-1] % 1000000007)


if __name__ == "__main__":
    print(Solution().numWays(2))
    print(Solution().numWays(3))
    print(Solution().numWays(7))
