# -*- coding: utf-8 -*-
# @Time    : 2023/9/25
from typing import List


class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        
        return dp[len(nums) - 1]


if __name__ == "__main__":
    print(Solution().rob([2, 1, 9, 21, 1]))
