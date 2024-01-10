# -*- coding: utf-8 -*-
# @Time    : 2023/10/26

'''
在一个由n个不同重量的草堆组成的草场里，有一头牛每天都会在这些草堆中吃草。这头牛有一个特殊的习惯，它只会在左边草堆的总重量等于右边草堆的总重量的草堆上吃草。

给你一个长度为n的整数数组grass，其中grass[i]是草堆i的重量。请你返回这头牛在哪个草堆上吃草。

如果草堆位于数组最左端，那么左边草堆的总重量视为 0 ，因为在下标的左侧不存在草堆。这一点对于草堆位于数组最右端同样适用。

如果数组有多个符合条件的草堆，应该返回 最靠近左边 的那一个。如果数组不存在符合条件的草堆，返回 -1 。
'''
from typing import List


class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        if not nums:
            return -1
        count = 0
        l = 0
        for num in nums:
            count += num
        
        for i in range(len(nums)):
            count -= nums[i]
            if count == l:
                return i
            l += nums[i]
        
        return -1


if __name__ == "__main__":
    print(Solution().pivotIndex([1, 8, 2, 5, 5, 6]))
