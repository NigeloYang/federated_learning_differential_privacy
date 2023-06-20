# -*- coding: utf-8 -*-
# @Time    : 2023/6/19


class Solution:
    def maxValue(self, grid):
        # m = len(grid)
        # n = len(grid[0])
        # res = grid[m-1][n-1]
        # if m == 1 and n >= 1:
        #     return sum(grid[0])
        # elif m >= 1 and n == 1:
        #     temp = 0
        #     for i in range(m):
        #         temp += sum(grid[i])
        #     return temp
        # while m > 0 and n > 0:
        #     if grid[m-1][n-2] > grid[m-2][n-1]:
        #         res += grid[m-1][n-2]
        #         m -= 1
        #     else:
        #         res += grid[m-2][n-1]
        #         n -= 1
        # return res
        
        # 先遍历第一个行列式，然后动态规划
        m, n = len(grid), len(grid[0])
        for j in range(1, n):  # 初始化第一行
            grid[0][j] += grid[0][j - 1]
        for i in range(1, m):  # 初始化第一列
            grid[i][0] += grid[i - 1][0]
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += max(grid[i][j - 1], grid[i - 1][j])
        return grid[-1][-1]


if __name__ == "__main__":
    lists = [
        [1, 3],
        [1, 5],
    ]

    lists2 = [
        [1, 3, 1]
    ]
    lists3 = [
        [11],
        [1],
        [4]
    ]
    lists4 = [
        [1, 3, 1],
        [1, 5, 1],
        [4, 2, 1]
    ]
    print(Solution().maxValue(lists))
    print(Solution().maxValue(lists2))
    print(Solution().maxValue(lists3))
    print(Solution().maxValue(lists4))
    print(Solution().maxValue([[0]]))
