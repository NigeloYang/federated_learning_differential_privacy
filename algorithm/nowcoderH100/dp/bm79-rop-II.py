# -*- coding: utf-8 -*-
# @Time    : 2023/9/26
from typing import List


class Solution:
    def rob(self, nums: List[int]) -> int:
        def myrob(nums: List[int]):
            cur,pre = 0,0
            for num in nums:
                cur,pre = max(pre+num,cur),cur
            return cur
        return max(myrob(nums[:-1]),myrob(nums[1:])) if len(nums) != 1 else nums[0]
if __name__ == "__main__":
    print()
