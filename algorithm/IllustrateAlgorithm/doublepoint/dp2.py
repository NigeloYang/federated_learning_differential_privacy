# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。

'''
from typing import List


class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        odd, even = [], []
        for num in nums:
            if num % 2 == 0:
                even.append(num)
            else:
                odd.append(num)
        return odd + even
    
    def exchange2(self, nums: List[int]) -> List[int]:
        i, j = 0, len(nums) - 1
        while i < j:
            while i < j and nums[i] & 1 == 1: i += 1
            while i < j and nums[j] & 1 == 0: j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        return nums


if __name__ == "__main__":
    print(Solution().exchange([1, 2, 3, 4, 5]))
    print(Solution().exchange2([1, 2, 3, 4, 5]))
