# -*- coding: utf-8 -*-
# @Time    : 2023/10/27
from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix:
            return False
        l, r = len(matrix) - 1, 0
        
        while l >= 0 and r < len(matrix[0]):
            if matrix[l][r] < target:
                l -= 1
            elif matrix[l][r] > target:
                r += 1
            else:
                return True
        return False


if __name__ == "__main__":
    print(Solution().searchMatrix([[9, 8, 7], [6, 5, 4], [3, 2, 1]], 5))
    print(Solution().searchMatrix([[30, 29, 28, 27, 26, 25, 24, 23, 22, 21],
                                   [20, 19, 18, 17, 16, 15, 14, 13, 12, 11],
                                   [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]], 25))
    print(Solution().searchMatrix([[60, 34, 30, 23], [20, 16, 11, 10], [7, 5, 3, 1]], 13))
