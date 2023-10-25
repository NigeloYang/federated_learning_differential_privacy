# -*- coding: utf-8 -*-
# @Time    : 2023/9/28

class Solution:
    def minWindow(self, s: str, T: str) -> str:
        if len(T) > len(s):
            return ""
        need = {}
        for t in T:
            if t in need:
                need[t] += 1
            else:
                need[t] = 1
        needcount = len(need)
        l = 0
        r = 0
        minl = len(s) + 1
        res = ''
        while r < len(s):
            if s[r] in need:
                need[s[r]] -= 1
                if need[s[r]] == 0:
                    needcount -= 1
            r += 1
            while needcount == 0:
                if r - l < minl:
                    minl = r - l
                    res = s[l:r]
                if s[l] in need:
                    need[s[l]] += 1
                    if need[s[l]] > 0:
                        needcount += 1
                l += 1
        
        return res


if __name__ == "__main__":
    print(Solution().minWindow('a', 'aa'))
    print(Solution().minWindow(s="ADOBECODEBANC", T="ABC"))
