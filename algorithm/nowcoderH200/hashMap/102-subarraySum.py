# -*- coding: utf-8 -*-
# @Time    : 2023/10/31
from typing import List


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> List[List[int]]:
        if not nums:
            return []
        ans = []
        count = set()
        for l in range(len(nums)):
            nums_sum = 0
            for r in range(l, len(nums)):
                nums_sum += nums[r]
                if nums_sum == k:
                    subnums = nums[l:r + 1]
                    substr = tuple(subnums)
                    if substr not in count:
                        count.add(substr)
                        ans.append(subnums)
                elif k < 0:
                    break
        return ans


if __name__ == "__main__":
    print(Solution().subarraySum([1, 2, 3, 4, 5], 5))
