# -*- coding: utf-8 -*-
# @Time    : 2023/10/29
from typing import List


class Solution:
    def partition(self, s: str) -> List[List[str]]:
        if not s:
            return []
        res = []
        
        def isp(s, l, r):
            while l < r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True
        
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if isp(s, i, j):
                    res.append(s[i:j + 1])
        ans = []
        for v in res:
            if v not in ans:
                ans.append(v)
        
        return ans


if __name__ == "__main__":
    print(Solution().partition("xxy"))
    print(Solution().partition("aaa"))
    print(Solution().partition("abcba"))
