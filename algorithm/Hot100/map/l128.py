# -*- coding: utf-8 -*-
# @Time    : 2023/7/3

'''最长连续序列

给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
'''


class Solution:
    def longestConsecutive(self, nums) -> int:
        if not nums:
            return 0
        nums.sort()
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            if nums[i] - nums[i - 1] == 1:
                dp[i] = dp[i - 1] + 1
            elif nums[i] == nums[i-1]:
                dp[i] = dp[i-1]
        
        return max(dp)
        
    def longestConsecutive1(self, nums) -> int:
        if len(nums) == 0:
            return 0
        nums.sort()
        count = 1
        res = 1
        for i in range(1, len(nums)):
            if nums[i] - nums[i - 1] == 1:
                count += 1
            else:
                if count > res:
                    res = count
                else:
                    count = 1
        if count > res:
            res = count
        return res


if __name__ == "__main__":
    print(Solution().longestConsecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]))
    print(Solution().longestConsecutive([9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6]))
    print(Solution().longestConsecutive([]))
    print(Solution().longestConsecutive1([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]))
    print(Solution().longestConsecutive1([9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6]))
    print(Solution().longestConsecutive1([]))
