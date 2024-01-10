# -*- coding: utf-8 -*-
# @Time    : 2023/10/31
from typing import List


class Solution:
    def cow_permute(self, nums: List[int]) -> List[List[int]]:
        ans = []
        path = []
        visited = [False] * len(nums)
        
        def dfs(cur):
            if cur == len(nums):
                ans.append(path[:])
                return
            
            for i in range(len(nums) - 1, -1, -1):
                if not visited[i]:
                    visited[i] = True
                    path.append(nums[i])
                    dfs(cur + 1)
                    path.pop()
                    visited[i] = False
        
        dfs(0)
        return ans

    def cow_permute2(self, nums: List[int]) -> List[List[int]]:
        result = []
        n = len(nums)
        self.backtracking(nums, n, [], result)
        return result


    def backtracking(self, nums, n, path, result):
        if len(path) == n:
            result.append(path[:])
            return
        for i in range(n):
            if nums[i] in path:
                continue
            path.append(nums[i])
            self.backtracking(nums, n, path, result)
            path.pop()


if __name__ == "__main__":
    print(Solution().cow_permute([1, 2, 3]))
    print(Solution().cow_permute2([1, 2, 3]))
    print(Solution().cow_permute2([1, 1, 3]))
