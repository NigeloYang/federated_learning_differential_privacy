#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/5 10:32
# @File    : d8.py
# @Author  : Richard Yang
'''剑指 Offer 58 - II. 左旋转字符串
字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

示例 1：
输入: s = "abcdefg", k = 2
输出: "cdefgab"
示例 2：

输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"

限制：
1 <= k < s.length <= 10000
'''


class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        # return s[n:] + s[:n]

        # 方案2
        res = ""
        for i in range(n, n + len(s)):
            res += s[i % len(s)]
        return res
    
        # 方案3
        # res = ""
        # for i in range(n, len(s)):
        #     res += s[i]
        # for i in range(n):
        #     res += s[i]
        # return res

if __name__ == "__main__":
    print(Solution().reverseLeftWords("abcdefg", 2))
