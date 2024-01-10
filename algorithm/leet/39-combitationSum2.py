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
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            self.dfs(nums, i + 1, path, res, target - nums[i])
            path.pop(-1)
    
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        used = [False] * len(candidates)
        res = []
        candidates.sort()
        self.backtracking(candidates, target, 0, used, res, [])
        return res
    
    def backtracking(self, candidates, target, startIndex, used, res, path):
        if target == 0:
            res.append(path[:])
            return
        for i in range(startIndex, len(candidates)):
            if i > startIndex and candidates[i] == candidates[i - 1] and not used[i - 1]:
                continue
            if target - candidates[i] < 0:
                break
            path.append(candidates[i])
            used[i] = True
            self.backtracking(candidates, target - candidates[i], i + 1, used, res, path)
            used[i] = False
            path.pop()


if __name__ == "__main__":
    print(Solution().combinationSum(candidates=[10, 1, 2, 7, 6, 1, 5], target=8))
    print(Solution().combinationSum2(candidates=[10, 1, 2, 7, 6, 1, 5], target=8))
