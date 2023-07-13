# -*- coding: utf-8 -*-
# @Time    : 2023/6/25
from typing import List

'''剑指 Offer 03. 数组中重复的数字
找出数组中重复的数字。
在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，
但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
'''
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        dic = set()
        for num in nums:
            if num in dic:
                return num
            else:
                dic.add(num)
        
if __name__ == "__main__":
    print(Solution().findRepeatNumber([2, 3, 1, 0, 2, 5, 3]))
