# -*- coding: utf-8 -*-
# @Time    : 2023/9/14


class Solution:
    def minPathSum(self, matrix: List[List[int]]) -> int:
        if len(matrix) == 0:
            return 0
        if len(matrix[0]) == 0:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        
        dp = [[0] * n for _ in range(m)]
        
        dp[0][0] = matrix[0][0]
        
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + matrix[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + matrix[0][j]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + matrix[i][j]
        
        return dp[m - 1][n - 1]
    
if __name__ == "__main__":
    print()
