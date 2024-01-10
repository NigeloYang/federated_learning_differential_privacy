# -*- coding: utf-8 -*-
# @Time    : 2023/10/31
from typing import List


class Solution:
    def cowCombinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        n = len(candidates)
        self.backtracking(candidates, n, [], result, target, 0)
        
        return result
    
    def backtracking(self, nums, n, path, result, target, start):
        if target == 0:
            result.append(path[:])
            return
        for i in range(start, n):
            if nums[i] <= target:
                path.append(nums[i])
                self.backtracking(nums, n, path, result, target - nums[i], i + 1)
                path.pop()


if __name__ == "__main__":
    print(Solution().cowCombinationSum([1, 3, 5], 6))
    print(Solution().cowCombinationSum([1, 2, 3, 4, 6, 7], 8))
