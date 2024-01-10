# -*- coding: utf-8 -*-
# @Time    : 2023/10/25
from typing import List


class Solution:
    def nextPermutation(self, cows: List[int]) -> List[int]:
        if not cows:
            return []
        
        i = len(cows) - 2
        while i >= 0 and cows[i] <= cows[i + 1]:
            i -= 1
        if i >= 0:
            # j = len(cows) - 1
            # while j >= 0 and cows[i] <= cows[j]:
            #     j -= 1
            # cows[i], cows[j] = cows[j], cows[i]
            cows[i], cows[i+1] = cows[i+1], cows[i]
        temp = cows[i+1:]
        temp.reverse()
        cows[i + 1:] = temp
        return cows


if __name__ == "__main__":
    print(Solution().nextPermutation([3, 2, 1]))
    print(Solution().nextPermutation([1, 2, 3]))
    print(Solution().nextPermutation([9,10,8,7,6,1,5,3,2,4]))
    print(Solution().nextPermutation([1,1,5]))
