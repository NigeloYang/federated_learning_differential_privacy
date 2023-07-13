# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''剑指 Offer 50. 第一个只出现一次的字符
在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。'''


class Solution:
    def firstUniqChar(self, s: str) -> str:
        if not len(s):
            return ' '
        
        res = {}
        for i in s:
            if i in res:
                res[i] += 1
            else:
                res[i] = 1
        for i, v in res.items():
            if v == 1:
                return i
        return ' '

if __name__ == "__main__":
    s = "abaccdeff"
    print(Solution().firstUniqChar(s))
