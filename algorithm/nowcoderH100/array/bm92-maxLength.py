# -*- coding: utf-8 -*-
# @Time    : 2023/9/28
from typing import List


class Solution:
    def maxLength(self , arr: List[int]) -> int:
        l,r,maxl = 0,0,0
        count = set()
        while r < len(arr):
            if arr[r] in count:
                count.remove(arr[l])
                l += 1
            else:
                count.add(arr[r])
                r += 1
                maxl = max(maxl,len(count))
        return maxl
if __name__ == "__main__":
    print()
