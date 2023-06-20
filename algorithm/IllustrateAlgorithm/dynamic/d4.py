# -*- coding: utf-8 -*-
# @Time    : 2023/6/19


class Solution:
    def maxSubArray(self, nums):
        # 方案1 空间复杂度 O(1)
        # for i in range(1, len(nums)):
        #     nums[i] += max(nums[i - 1], 0)
        # return max(nums)
        
        # 方案2 空间复杂度 O(N)
        leng = len(nums)
        dp = [0] * leng
        dp[0] = nums[0]
        for i in range(1, leng):
            if dp[i - 1] <= 0:
                dp[i] = nums[i]
            else:
                dp[i] = dp[i - 1] + nums[i]
        return max(dp)


if __name__ == "__main__":
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(Solution().maxSubArray(nums))
