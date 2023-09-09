# -*- coding: utf-8 -*-
# @Time    : 2023/7/26

'''岛屿数量
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。'''
from typing import List


class Solution:
    # 广度优先遍历
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == '1':  # 发现陆地
                    count += 1  # 结果加1
                    grid[row][col] = '0'  # 将其转为 ‘0’ 代表已经访问过
                    # 对发现的陆地进行扩张即执行 BFS，将与其相邻的陆地都标记为已访问
                    land_positions = []
                    land_positions.append([row, col])
                    while len(land_positions) > 0:
                        x, y = land_positions.pop(0)
                        for new_x, new_y in [[x, y + 1], [x, y - 1], [x + 1, y], [x - 1, y]]:  # 进行四个方向的扩张
                            # 判断有效性
                            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] == '1':
                                grid[new_x][new_y] = '0'  # 因为可由 BFS 访问到，代表同属一块岛，将其置 ‘0’ 代表已访问过
                                land_positions.append([new_x, new_y])
        return count
    
    # 深度优先遍历
    def numIslands2(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        row = len(grid)
        col = len(grid[0])
        count = 0
    
        for i in range(row):
            for j in range(col):
                if grid[i][j] == '1':
                    self.dfs(grid, i, j)
                    count += 1
        return count

    def dfs(self, grid, i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1':
            return
        grid[i][j] = '0'  # 将其转为 ‘0’ 代表已经访问过
        self.dfs(grid, i + 1, j)
        self.dfs(grid, i - 1, j)
        self.dfs(grid, i, j + 1)
        self.dfs(grid, i, j - 1)
    
    def numIslands3(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        row = len(grid)
        col = len(grid[0])
        count = 0
        def dfs(i, j):
            grid[i][j] = '0'  # 将其转为 ‘0’ 代表已经访问过
            for ni, nj in [[i + 1, j], [i - 1, j], [i, j + 1], [i, j + 1]]:
                if 0 <= ni < row and 0 <= nj < col and grid[ni][nj] == '1':
                    dfs( ni, nj)
        for i in range(row):
            for j in range(col):
                if grid[i][j] == '1':
                    dfs(i, j)
                    count += 1
        return count


if __name__ == "__main__":
    grid = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
    ]
    print(Solution().numIslands(grid))
    print(Solution().numIslands2(grid))
    print(Solution().numIslands([["1","1","1"],["0","1","0"],["1","1","1"]]))
    print(Solution().numIslands2([["1","1","1"],["0","1","0"],["1","1","1"]]))
    print(Solution().numIslands3([["1","1","1"],["0","1","0"],["1","1","1"]]))
