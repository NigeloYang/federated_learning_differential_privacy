# -*- coding: utf-8 -*-
# @Time    : 2023/9/5

class Solution:
    def findPeakElement(self , nums: List[int]) -> int:
        l = 0
        r = len(nums) - 1

        while l < r:
            m = int((l+r)/2)
            if nums[m] > nums[m+1]:
                r = m
            else:
                l = m + 1
        return r
    
if __name__ == "__main__":
    print()
