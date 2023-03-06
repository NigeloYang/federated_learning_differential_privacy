#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/5 10:39
# @File    : d9.py
# @Author  : Richard Yang
'''剑指 Offer 59 - I. 滑动窗口的最大值
给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。

示例:

输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7]
解释:

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
 

提示：

你可以假设 k 总是有效的，在输入数组 不为空 的情况下，1 ≤ k ≤ nums.length。

注意：本题与主站 239 题相同：
'''
import collections
from typing import List


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 多数组的情况下，时间超出限制
        # if not nums:
        #     return []
        #
        # res = []
        # for i in range(len(nums) - k + 1):
        #     res.append(max(nums[i:i + k]))

        if not nums or k == 0:
            return []
        
        deque = collections.deque()
        # 未形成窗口
        for i in range(k):
            while deque and deque[-1] < nums[i]:
                deque.pop()
            deque.append(nums[i])
        res = [deque[0]]
        print(deque)
        # 形成窗口后
        for i in range(k, len(nums)):
            if deque[0] == nums[i - k]:
                deque.popleft()
            while deque and deque[-1] < nums[i]:
                deque.pop()
            deque.append(nums[i])
            res.append(deque[0])
        return res


if __name__ == "__main__":
    print(Solution().maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))
    # print(Solution().maxSlidingWindow([1], 1))
