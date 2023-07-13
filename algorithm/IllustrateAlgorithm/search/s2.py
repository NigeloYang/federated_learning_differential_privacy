# -*- coding: utf-8 -*-
# @Time    : 2023/6/23

'''剑指 Offer 04. 二维数组中的查找
在一个 n * m 的二维数组中，每一行都按照从左到右 非递减 的顺序排序，每一列都按照从上到下 非递减 的顺序排序。
请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
'''
from typing import List


class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        # start_l, start_r, end_l, end_r = 0, 0, len(matrix) , len(matrix[0])
        # while start_l <= end_l:
        #     mid1 = (start_l + end_l) // 2
        #     if matrix[mid1][start_r] < target:
        #         start_l = mid1 + 1
        #     elif matrix[mid1][start_r] > target:
        #         end_l = mid1 - 1
        # while start_r <= end_r:
        #     mid2 = (start_r + end_r) // 2
        #     if matrix[mid1][mid2] < target:
        #         start_r = mid2 + 1
        #     elif matrix[mid1][mid2] > target:
        #         end_r = mid2 - 1
        # if (mid1 == len(matrix) and mid2 == len(matrix[0])) or matrix[mid1][mid2] != target:
        #     return False
        # return True
        row, col = len(matrix)-1, 0
        while row >= 0 and col < len(matrix[0]):
            if matrix[row][col] > target:
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else:
                return True
        return False


if __name__ == "__main__":
    matrix = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30]
    ]
    target = 5
    print(Solution().findNumberIn2DArray(matrix, target))
