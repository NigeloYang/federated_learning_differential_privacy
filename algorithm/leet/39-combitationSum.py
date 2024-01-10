# -*- coding: utf-8 -*-
# @Time    : 2023/12/8
from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        if not candidates or (len(candidates) == 1 and candidates[0] > target):
            return []
        candidates.sort()
        res = []
        path = []
        self.dfs(candidates, 0, path, res, target)
        return res
    
    def dfs(self, nums, start, path, res, target):
        if target == 0:
            res.append(list(path))
            return
        for i in range(start, len(nums)):
            if target - nums[i] < 0:
                break
            path.append(nums[i])
            self.dfs(nums, i, path, res, target - nums[i])
            path.pop(-1)


if __name__ == "__main__":
    print(Solution().combinationSum([2, 3, 6, 7], 7))
