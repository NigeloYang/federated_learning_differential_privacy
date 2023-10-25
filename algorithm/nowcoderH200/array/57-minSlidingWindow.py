# -*- coding: utf-8 -*-
# @Time    : 2023/10/24
import collections
from typing import List


class Solution:
    def minSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if k == 1:
            return nums
        res = []
        queue = collections.deque()
        for i in range(len(nums)):
            while queue and queue[0] == i - k:
                queue.popleft()
            
            while queue and nums[queue[-1]] > nums[i]:
                queue.pop()
            
            queue.append(i)
            
            if i >= k - 1:
                res.append(nums[queue[0]])
        return res


if __name__ == "__main__":
    print(Solution().minSlidingWindow([1, 2, 3, 4, 5, 6], 3))
    print(Solution().minSlidingWindow([-1, -2, -3, -4, -5, -6], 2))
