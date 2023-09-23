# -*- coding: utf-8 -*-
# @Time    : 2023/9/17

class Solution:
    def numIslandsDFS(self, grid: List[List[str]]) -> int:
        def dfs(grid, i, j):
            if not 0 <= i < len(grid) or not 0 <= j < len(grid[0]) or grid[i][j] == '0':
                return
            for ni, nj in [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]:
                dfs(grid, ni, nj)
        
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(grid, i, j)
                    count += 1
        return count
    
    def numIslandsBFS(self, grid: List[List[str]]) -> int:
        def bfs(grid, i, j):
            que = [[i, j]]
            while que:
                [i, j] = que.pop(0)
                if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == '1':
                    grid[i][j] = '0'
                    que += [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]
        
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '0':
                    continue
                bfs(grid, i, j)
                count += 1
        return count


if __name__ == "__main__":
    print()
