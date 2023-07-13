# -*- coding: utf-8 -*-
# @Time    : 2023/7/3

'''找到字符串中所有字母异位词
给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。'''


class Solution:
    def findAnagrams(self, s: str, p: str):
        s1 = sorted(p)
        n = len(p)
        res = []
        for i in range(n, len(s) + 1):
            temp = sorted(s[i - n:i])
            if s1 == temp:
                res.append(i - n)
        return res


if __name__ == "__main__":
    s = "cbaebabacd"
    p = "abc"
    print(Solution().findAnagrams(s, p))
    print(Solution().findAnagrams('abab', 'ab'))
