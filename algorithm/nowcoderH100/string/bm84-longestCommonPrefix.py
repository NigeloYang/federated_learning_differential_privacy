# -*- coding: utf-8 -*-
# @Time    : 2023/9/27
from typing import List


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        
        m = len(strs)
        n = len(strs[0])
        res = strs[0]
        for strt in strs:
            if len(strt) < n:
                n = len(strt)
                res = strt
        
        while n > 0:
            temp = 0
            for strt in strs:
                if res[:n] == strt[:n]:
                    temp += 1
                else:
                    temp -= 1
            if temp == m:
                return res[:n]
            n -= 1
        return ""


if __name__ == "__main__":
    print(Solution().longestCommonPrefix([""]))
    print(Solution().longestCommonPrefix(["a", 'b']))
