# -*- coding: utf-8 -*-
# @Time    : 2023/7/7

'''搜索二维矩阵 II
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。'''


class Solution:
    def searchMatrix(self, matrix, target: int) -> bool:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == target:
                    return True
        return False
    
    def searchMatrix2(self, matrix, target: int) -> bool:
        # Z 字形查找
        n = len(matrix)
        i, j = 0, len(matrix[0])-1
        while i < n and j >= 0:
            if matrix[i][j] == target:
                return True
            if matrix[i][j] > target:
                j -= 1
            else:
                i += 1
        return False
    def searchMatrix3(self, matrix, target: int) -> bool:
        # 二分查找
        n = len(matrix)
        i, j = 0, len(matrix[0])-1
        while i < n and j >= 0:
            if matrix[i][j] == target:
                return True
            if matrix[i][j] > target:
                j -= 1
            else:
                i += 1
        return False


if __name__ == "__main__":
    matrix = [[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24],
              [18, 21, 23, 26, 30]]
    target = 5
    
    print(Solution().searchMatrix(matrix, target))
    print(Solution().searchMatrix2(matrix, target))
    print(Solution().searchMatrix3(matrix, target))
