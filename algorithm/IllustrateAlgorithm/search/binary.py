# -*- coding: utf-8 -*-
# @Time    : 2023/6/23

'''二分查找
是一种基于分治思想的高效搜索算法。它利用数据的有序性，每轮减少一半搜索范围，直至找到目标元素或搜索区间为空为止
'''
from typing import List


class Solution:
    # 无重复
    def binary_search(self, nums: List[int], target: int) -> int:
        r = len(nums) - 1
        l = 0
        while l <= r:
            m = (l + r) // 2
            if nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
            else:
                return m
        return -1
    
    # 有重复，需要边界查找
    def binary_search_left_edge(self, nums: List[int], target: int) -> int:
        r = len(nums) - 1
        l = 0
        while l <= r:
            m = (l + r) // 2
            if nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
            else:
                r = m - 1
        if l == len(nums) or nums[l] != target:
            return -1
        return l
    
    def binary_search_right_edge(self, nums: List[int], target: int) -> int:
        r = len(nums) - 1
        l = 0
        while l <= r:
            m = (l + r) // 2
            if nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
            else:
                l = m + 1
        if r == len(nums) or nums[r] != target:
            return -1
        return r


if __name__ == "__main__":
    print(Solution().binary_search([1, 4, 7, 11, 15], 11))
    print(Solution().binary_search_left_edge([1, 4, 3, 6, 7, 7, 7, 7, 7, 7, 11, 15], 7))
    print(Solution().binary_search_right_edge([1, 4, 3, 6, 7, 7, 7, 7, 7, 7, 11, 15], 7))
