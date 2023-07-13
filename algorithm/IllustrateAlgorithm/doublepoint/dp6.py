# -*- coding: utf-8 -*-
# @Time    : 2023/6/26


'''剑指 Offer 57. 和为 s 的两个数字
输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
'''
from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        res = set()
        for num in nums:
            if target - num in res:
                return [target - num, num]
            else:
                res.add(num)
        return []
    
    def twoSums2(self, nums: List[int], target: int) -> List[int]:
        l, r = 0, len(nums) - 1
        while l < r:
            if nums[l] + nums[r] > target:
                r -= 1
            elif nums[l] + nums[r] < target:
                l += 1
            else:
                return [nums[l], nums[r]]
        return []


if __name__ == "__main__":
    nums = [10, 26, 30, 31, 47, 60]
    target = 40
    print(Solution().twoSums2(nums,target))
