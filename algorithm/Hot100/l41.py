# -*- coding: utf-8 -*-
# @Time    : 2023/7/6

''' 缺失的第一个正数
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。'''


class Solution:
    def firstMissingPositive(self, nums) -> int:
        if not nums:
            return 1
        nums.sort()
        temp = 0
        for i in range(len(nums)):
            if nums[i] > 0:
                if nums[i] - temp == 1:
                    temp += 1
                elif nums[i] - temp == 0:
                    temp = temp
                else:
                    return temp + 1
        
        return 1 if nums[-1] < 0 else nums[-1] + 1


if __name__ == "__main__":
    print(Solution().firstMissingPositive([]))
    print(Solution().firstMissingPositive([1, 2, 0]))
    print(Solution().firstMissingPositive([3, 4, -1, 1]))
    print(Solution().firstMissingPositive([7, 8, 9, 11, 12]))
    print(Solution().firstMissingPositive([-5, -4, -3]))
    print(Solution().firstMissingPositive([0, 2, 2, 1, 1]))
