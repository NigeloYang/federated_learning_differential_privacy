# -*- coding: utf-8 -*-
# @Time    : 2023/12/1
from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1
        if len(nums) == 1 and nums[0] != target:
            return -1
        
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + (r - l) // 2
            if nums[m] == target:
                return m
            if nums[l] <= nums[m]:
                if target >= nums[l] and target <= nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:
                if target >= nums[m] and target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
        
        return -1

    def search2(self, nums: List[int], target: int) -> int:
        # 思路：二分查找-左右部分分别使用-逐渐缩小边界
        # 区分左右边界哪边有序
        # 分左右部分缩小边界
        
        left, right = 0, len(nums) - 1
        
        while left <= right:
            
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[left] <= nums[mid]:
                if nums[left] <= target <= nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            
            if nums[mid] <= nums[right]:
                if nums[mid] <= target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1


if __name__ == "__main__":
    print(Solution().search([4, 5, 6, 7, 0, 1, 2], 0))
    print(Solution().search2([4, 5, 6, 7, 0, 1, 2], 0))
