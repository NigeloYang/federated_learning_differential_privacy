# -*- coding: utf-8 -*-
# @Time    : 2023/12/1
from typing import List


class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # # 找到要更换的索引位置
        # i = len(nums) - 2
        # while i >= 0 and nums[i] >= nums[i + 1]:
        #     i -= 1
        # # 执行交换
        # if i >= 0:
        #     j = len(nums) - 1
        #     while j >= 0 and nums[i] >= nums[j]:
        #         j -= 1
        #     nums[i], nums[j] = nums[j], nums[i]
        #
        # # 变换后面的顺序
        # l, r = i + 1, len(nums) - 1
        # while l < r:
        #     nums[l], nums[r] = nums[r], nums[l]
        #     l += 1
        #     r -= 1
        #
        
        i, j, k = len(nums) - 2, len(nums) - 1, len(nums) - 1
        while i >= 0 and nums[i] >= nums[j]:
            i -= 1
            j -= 1
            
        if i >= 0:
            while nums[i] >= nums[k]:
                k -= 1
            nums[i], nums[k] = nums[k], nums[i]
        
        left, right = j, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
        
        return nums


if __name__ == "__main__":
    print(Solution().nextPermutation([1, 2, 3]))
    print(Solution().nextPermutation([3, 2, 1]))
