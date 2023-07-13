# -*- coding: utf-8 -*-
# @Time    : 2023/6/23

'''我们常通过将线性查找替换为哈希查找来降低算法的时间复杂度'''
from typing import List


class Solution:
    # 无重复
    def two_sum_hash_table(self, nums: List[int], target: int) -> int:
        dic = {}
        
        for i in range(len(nums)):
            if target - nums[i] in dic:
                return [dic[target - nums[i]], i]
            dic[nums[i]] = i
        return []


if __name__ == "__main__":
    print(Solution().two_sum_hash_table([1, 4, 7, 11, 15], 8))
