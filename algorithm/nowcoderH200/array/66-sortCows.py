# -*- coding: utf-8 -*-
# @Time    : 2023/10/25
from typing import List


class Solution:
    def sortCows(self, cows: List[int]) -> List[int]:
        if not cows:
            return []
        
        count1 = []
        count2 = []
        for cow in cows:
            if cow == 0:
                count1.append(cow)
            else:
                count2.append(cow)
        count1.extend(count2)
        return count1
    
    def sortCows2(self, cows: List[int]) -> List[int]:
        if not cows:
            return []
        
        s = 0
        for i in range(len(cows)):
            if cows[i] == 0:
                cows[s] = 0
                s += 1
        
        while s < len(cows):
            cows[s] = 1
            s += 1
        
        return cows


if __name__ == "__main__":
    print(Solution().sortCows2([1, 0, 1, 0, 1, 0]))
