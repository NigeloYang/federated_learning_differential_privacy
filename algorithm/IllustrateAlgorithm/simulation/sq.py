# -*- coding: utf-8 -*-
# @Time    : 2023/6/26

'''剑指 Offer 29. 顺时针打印矩阵
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。'''
from typing import List


class Solution:
    def spiralOrder(self, matrix):
        if not matrix:
            return []
        l, t, r, b, res = 0, 0, len(matrix[0]) - 1, len(matrix) - 1, []
        while True:
            # 从左到右
            for i in range(l, r + 1):
                res.append(matrix[t][i])
            t += 1
            if t > b: break
            
            # 从上到下
            for i in range(t, b + 1):
                res.append(matrix[i][r])
            r -= 1
            if r < l: break
            
            # 从右到左
            for i in range(r, l - 1, -1):
                res.append(matrix[b][i])
            b -= 1
            if b < t: break
            
            # 从下到上
            for i in range(b, t - 1, -1):
                res.append(matrix[i][l])
            l += 1
            if l > r: break
        return res


if __name__ == "__main__":
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    print(Solution().spiralOrder(matrix))
    print(Solution().spiralOrder2(matrix))
