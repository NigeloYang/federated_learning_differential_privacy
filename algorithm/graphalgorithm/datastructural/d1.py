#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 19:50
# @File    : d1.py
# @Author  : Richard Yang
'''剑指 Offer 05. 替换空格
请实现一个函数，把字符串 s 中的每个空格替换成"%20"。


示例 1：

输入：s = "We are happy."
输出："We%20are%20happy."

限制：
0 <= s 的长度 <= 10000
'''


class Solution:
    def replaceSpace(self, s: str) -> str:
        # 方案1
        # return s.replace(' ', '%20')

        # 方案2
        res = []
        for i in s:
            if i == " ":
                res.append("%20")
            else:
                res.append(i)
        return ''.join(res)


if __name__ == "__main__":
    print(Solution().replaceSpace("We are happy."))
