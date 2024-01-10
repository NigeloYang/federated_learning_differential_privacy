# -*- coding: utf-8 -*-
# @Time    : 2023/10/25
from typing import List


class Solution:
    def sortCows(self, cows: List[int]) -> List[int]:
        if not cows:
            return []
        
        count = [0] * 3
        for cow in cows:
            count[cow] += 1
        ans = []
        for k, v in enumerate(count):
            ans.extend([k] * v)
        
        return ans
    
    def sortCows2(self, cows: List[int]) -> List[int]:
        if not cows:
            return []
        
        l, r = 0, len(cows) - 1
        ans = [0] * len(cows)
        for i in range(len(cows)):
            if cows[i] == 0:
                l += 1
            elif cows[i] == 2:
                ans[r] = 2
                r -= 1
        
        while l <= r:
            ans[l] = 1
            l += 1
        return ans


if __name__ == "__main__":
    print(Solution().sortCows2([0,0,0,1,2]))
