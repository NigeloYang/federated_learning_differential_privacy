# -*- coding: utf-8 -*-
# @Time    : 2023/9/30

class Solution:
    def rotateMatrix(self , mat: List[List[int]], n: int) -> List[List[int]]:
        if n == 0:
            return mat
        new_mat = [[0] * n for i in range(n)]
        for i in range(n):
            for j in range(n):
                new_mat[j][n-1-i] = mat[i][j]
        mat[:] = new_mat[:]
        return mat
    
if __name__ == "__main__":
    print()
