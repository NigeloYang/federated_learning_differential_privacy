# -*- coding: utf-8 -*-
# @Time    : 2023/7/6
'''矩阵置零
给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。'''

class Solution:
    def setZeroes(self, matrix) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        m = len(matrix[0])
        pos = [[False] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    pos[i][j] = True
        for i in range(n):
            for j in range(m):
                if pos[i][j] == True:
                    for ij in range(n):
                        matrix[ij][j] = 0
                    for ij in range(m):
                        matrix[i][ij] = 0
        
                    
                    
if __name__ == "__main__":
    matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    print(Solution().setZeroes(matrix))
