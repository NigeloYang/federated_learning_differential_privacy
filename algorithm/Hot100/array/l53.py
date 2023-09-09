# -*- coding: utf-8 -*-
# @Time    : 2023/7/26

''' 最大子数组和
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
子数组 是数组中的一个连续部分。

理解动态规划：https://leetcode.cn/problems/maximum-subarray/solutions/9058/dong-tai-gui-hua-fen-zhi-fa-python-dai-ma-java-dai/
'''


class Solution:
    def maxSubArray(self, nums) -> int:
        n = len(nums)
        res, pre = nums[0], nums[0]
        for i in range(1, n):
            if pre > 0:
                pre = pre + nums[i]
            else:
                pre = nums[i]
            res = max(res, pre)
        return res
    
    def maxSubArray2(self, nums) -> int:
        if not nums:
            return 0
        
        dp = [0 for i in range(len(nums))]
        dp[0] = nums[0]
        for i in range(1,len(nums)):
            if dp[i-1] > 0:
                dp[i] = dp[i-1] + nums[i]
            else:
                dp[i] = nums[i]
        print(dp)
        return max(dp)
        


if __name__ == "__main__":
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(Solution().maxSubArray(nums))
    print(Solution().maxSubArray2(nums))
    print(Solution().maxSubArray2([4,-1,-1,-1,4,2,5]))
