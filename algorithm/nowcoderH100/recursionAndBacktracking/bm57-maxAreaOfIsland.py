# -*- coding: utf-8 -*-
# @Time    : 2023/9/17
import collections
from typing import List


class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                ans = max(self.dfs(grid, i, j), ans)
        return ans
    
    def dfs(self, grid, i, j):
        if not 0 <= i < len(grid) or not 0 <= j < len(grid[0]) or grid[i][j] == 0:
            return 0
        area = 1
        grid[i][j] = 0
        for ni, nj in [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]:
            area += self.dfs(grid, ni, nj)
        return area
    
    def maxAreaOfIsland2(self, grid: List[List[int]]) -> int:
        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                ans = max(self.bfs(grid, i, j), ans)
        return ans
    
    def bfs(self, grid, i, j):
        que = collections.deque([[i,j]])
        area = 0
        while que:
            i, j = que.popleft()
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == 1:
                area += 1
                grid[i][j] = 0
                que += [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]

        return area

if __name__ == "__main__":
    print(Solution().maxAreaOfIsland2([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                                      [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]))
