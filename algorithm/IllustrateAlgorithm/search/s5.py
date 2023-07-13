# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''剑指 Offer 53 - I. 在排序数组中查找数字 I
统计一个数字在排序数组中出现的次数。'''
from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        count = 0
        for num in nums:
            if num == target:
                count += 1
            elif num > target:
                break
        return count
    
    def search2(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            return (nums[mid] == target) + self.search(nums[0:mid], target) + self.search(nums[mid + 1:], target)
        
        return 0
    
    def search3(self, nums: List[int], target: int) -> int:
        count = 0
        s = nums.index(target)
        for i in range(s, len(nums)):
            if nums[i] == target:
                count += 1
            elif nums[i] > target:
                break
        return count
    
    def search4(self, nums: [int], target: int) -> int:
        # 搜索右边界 right
        i, j = 0, len(nums) - 1
        while i <= j:
            m = (i + j) // 2
            if nums[m] <= target:
                i = m + 1
            else:
                j = m - 1
        right = i
        # 若数组中无 target ，则提前返回
        if j >= 0 and nums[j] != target: return 0
        # 搜索左边界 left
        i = 0
        while i <= j:
            m = (i + j) // 2
            if nums[m] < target:
                i = m + 1
            else:
                j = m - 1
        left = j
        return right - left - 1


if __name__ == "__main__":
    print(Solution().search([5, 7, 7, 8, 8, 10], 8))
    print(Solution().search2([5, 7, 7, 8, 8, 10], 8))
    print(Solution().search3([5, 7, 7, 8, 8, 10], 8))
    print(Solution().search4([5, 7, 7, 8, 8, 10], 8))
