# -*- coding: utf-8 -*-
# @Time    : 2023/9/6

class Solution:
    def minNumberInRotateArray(self , nums: List[int]) -> int:
        l = 0
        r = len(nums) -1
        while l < r:
            m = int((l+r)/2)
            if nums[m] > nums[r]:
                l = m +1
            elif nums[m] == nums[r]:
                r -= 1
            else:
                r = m
        return nums[l]
    
if __name__ == "__main__":
    print()
