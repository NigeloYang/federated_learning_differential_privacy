# -*- coding: utf-8 -*-
# @Time    : 2023/7/4

'''最小覆盖子串
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：
对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
如果 s 中存在这样的子串，我们保证它是唯一的答案。'''
import collections


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.defaultdict(int)
        for k in t:
            need[k] += 1
        needcount = len(t)
        l = 0
        res = (0, float('inf'))
        for r, k in enumerate(s):
            if need[k] > 0:
                needcount -= 1
            need[k] -= 1
            print(need)
            if needcount == 0:  # 步骤一：滑动窗口包含了所有T元素
                print('input needcount')
                while True:  # 步骤二：增加l，排除多余元素
                    k = s[l]
                    if need[k] == 0:
                        break
                    else:
                        need[k] += 1
                        l += 1
                    print(need)
                if r - l < res[1] - res[0]:  # 记录结果
                    res = (l, r)
                need[s[l]] += 1  # 步骤三：i增加一个位置，寻找新的满足条件滑动窗口
                needcount += 1
                l += 1
                print('end needcount')
        return "" if res[1] > len(s) else s[res[0]:res[1] + 1]


if __name__ == "__main__":
    s = "AADOBECODEBANC"
    t = "ABC"
    print(Solution().minWindow(s, t))
