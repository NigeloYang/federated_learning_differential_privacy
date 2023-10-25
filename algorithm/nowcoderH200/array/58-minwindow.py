# -*- coding: utf-8 -*-
# @Time    : 2023/10/24

import collections

class Solution:
    def minWindow(self , s: str, t: str) -> str:
        need = collections.defaultdict(int)
        # 统计查询的字串的字母数量
        for v in t:
            need[v] += 1
        needcount = len(t)
        l = 0
        res = (0, float('inf'))
        for r, v in enumerate(s):
            if need[v] > 0:
                needcount -= 1
            need[v] -= 1
            if needcount == 0:  # 步骤一：滑动窗口包含了所有T元素
                while True:  # 步骤二：左边索引 l 向右增加，排除多余元素
                    if need[s[l]] == 0:
                        break
                    else:
                        need[s[l]] += 1
                        l += 1
                if r - l < res[1] - res[0]:  # 记录新的滑动窗口大小
                    res = (l, r)
                need[s[l]] += 1  # 步骤三：T中的内容增加一个位置，以寻找新的满足条件滑动窗口
                needcount += 1
                l += 1
        return "" if res[1] > len(s) else s[res[0]:res[1] + 1]
    
if __name__ == "__main__":
    print(Solution().minWindow("ABCDEFGHIJKLMNOPQRSTUVWXYZ","XZ"))
