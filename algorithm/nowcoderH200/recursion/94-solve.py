# -*- coding: utf-8 -*-
# @Time    : 2023/10/31
from typing import List

'''
在一个牧场中，有很多牛。为了方便管理，牧场主将牛的编号排列成一个 m x n 的矩阵。矩阵中的每个元素表示一个牛的位置，'A' 表示有牛，'B' 表示没有牛。请你编写一个程序，找到所有被 'A' 围绕的区域，并将这些区域里所有的 'B' 用 'A' 填充。

被围绕的区间不会存在于边界上，换句话说，任何边界上的 'B' 都不会被填充为 'A'。任何不在边界上，或不与边界上的 'B' 相连的 'B' 最终都会被填充为 'A'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
'''


class Solution:
    def solve(self, board: List[List[str]]) -> List[List[str]]:
        if not board:
            return []
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'B':
                    self.dfs(board, i, j)
        
        return board
    
    def dfs(self, nums: List[List[str]], i: int, j: int):
        if i <= 0 or i >= len(nums) - 1 or j <= 0 or j >= len(nums[0]) - 1 or nums[i][j] == 'A':
            return
        
        nums[i][j] = 'A'
        self.dfs(nums, i + 1, j)
        self.dfs(nums, i - 1, j)
        self.dfs(nums, i, j + 1)
        self.dfs(nums, i, j - 1)


if __name__ == "__main__":
    print(Solution().solve([['A', 'A', 'A', 'A'], ['A', 'B', 'B', 'A'], ['A', 'A', 'B', 'A'], ['A', "B", 'A', 'A']]))
