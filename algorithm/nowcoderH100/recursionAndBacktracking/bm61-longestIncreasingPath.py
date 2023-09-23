# -*- coding: utf-8 -*-
# @Time    : 2023/9/23
from typing import List


class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return
        if not matrix[0]:
            return
        m, n = len(matrix), len(matrix[0])
        ans = 0
        visited = [[0] * n for i in range(m)]
        for i in range(m):
            for j in range(n):
                ans = max(ans, self.dfs(matrix, i, j, visited))
        return ans

    def dfs(self, matrix: List[List[int]], i: int, j: int, visited: List[List[int]]):
        if visited[i][j] != 0:
            return visited[i][j]
    
        visited[i][j] += 1
        for ni, nj in [[i + 1, j], [i - 1, j], [i, j + 1], [i, j - 1]]:
            if 0 <= ni < len(matrix) and 0 <= nj < len(matrix[0]) and matrix[ni][nj] > matrix[i][j]:
                visited[i][j] = max(visited[i][j], self.dfs(matrix, ni, nj, visited) + 1)
        return visited[i][j]


if __name__ == "__main__":
    print(Solution().longestIncreasingPath([[3, 4, 5], [3, 2, 6], [2, 2, 1]]))
