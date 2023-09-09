# -*- coding: utf-8 -*-
# @Time    : 2023/7/26
'''01 矩阵
给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。
两个相邻元素间的距离为 1 。'''

from typing import List


class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        res = [[None for i in range(len(mat[0]))] for i in range(len(mat))]
        print(res)
        visited = [] # BFS 经典结果，设定一个 queue 来存储每个层次上的点
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] == 0:
                    res[i][j] = 0    # 0到自身的距离为零
                    visited.append([i, j]) # 将找到的 0 放入队列
        while visited:
            x, y = visited.pop(0)
            for x_bias, y_bias in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                new_x = x + x_bias
                new_y = y + y_bias
                if 0 <= new_x < len(mat) and 0 <= new_y < len(mat[0]) and res[new_x][new_y] == None:
                    res[new_x][new_y] = res[x][y] + 1
                    visited.append([new_x, new_y]) # 将新扩展的点加入队列
        return res


if __name__ == "__main__":
    mat = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    mat2 = [[0, 0, 0], [0, 1, 0], [1, 1, 1]]
    print(Solution().updateMatrix(mat))
    print(Solution().updateMatrix(mat2))
