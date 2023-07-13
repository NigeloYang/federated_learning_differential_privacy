# -*- coding: utf-8 -*-
# @Time    : 2023/7/3

''' 和为 K 的子数组
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的连续子数组的个数 。'''
from typing import List


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # 前缀和 + 字典
        dic = {0: 1}
        sums, res = 0, 0
        
        for num in nums:
            sums += num
            res += dic.get(sums - k, 0)
            dic[sums] = dic.get(sums, 0) + 1
        return res
    
    def subarraySum2(self, nums: List[int], k: int) -> int:
        # 枚举
        res = 0
        for i in range(len(nums)):
            sums = 0
            for j in range(i, -1, -1):
                sums += nums[j]
                if sums == k:
                    res += 1
        return res


if __name__ == "__main__":
    print(Solution().subarraySum([1, 1, 1], 3))
    print(Solution().subarraySum2([1, 1, 1], 3))
    print(Solution().subarraySum([1, 2, 3, 4, 5, 6, 7,11], 11))
    print(Solution().subarraySum2([1, 2, 3, 4, 5, 6, 7,11],11))
