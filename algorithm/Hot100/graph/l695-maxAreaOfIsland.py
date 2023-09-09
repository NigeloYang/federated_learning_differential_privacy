# -*- coding: utf-8 -*-
# @Time    : 2023/7/26

'''岛屿的最大面积'''


class Solution:
    def maxAreaOfIsland(self, grid) -> int:
        area = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                temparea = 0
                if grid[row][col] == 1:  # 发现陆地
                    temparea += 1  # 结果加1
                    grid[row][col] = 0  # 将其转为 ‘0’ 代表已经访问过, 对发现的陆地进行扩张即执行 BFS，将与其相邻的陆地都标记为已访问
                    land_positions = [[row, col]]
                    while len(land_positions) > 0:
                        x, y = land_positions.pop(0)
                        for new_x, new_y in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:  # 进行四个方向的扩张
                            # 判断有效性
                            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] == 1:
                                temparea += 1
                                grid[new_x][new_y] = 0  # 因为可由 BFS 访问到，代表同属一块岛，将其置 ‘0’ 代表已访问过
                                land_positions.append([new_x, new_y])
                    
                    area = max(area, temparea)
        return area
    
    def maxAreaOfIsland2(self, grid) -> int:
        area = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == 1:  # 发现陆地
                    area = max(self.dfs(grid, row, col), area)
        return area
    
    def dfs(self, grid, i, j):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != 1:
            return 0
        area = 1
        grid[i][j] = 0
        for new_x, new_y in [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]:  # 进行四个方向的扩张
            area += self.dfs(grid, new_x, new_y)
        return area


if __name__ == "__main__":
    print(Solution().maxAreaOfIsland([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                                      [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]))
    print(Solution().maxAreaOfIsland2([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                                      [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]))
    
    print(Solution().maxAreaOfIsland3([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                                      [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]))