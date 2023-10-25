# -*- coding: utf-8 -*-
# @Time    : 2023/9/30

'''螺旋矩阵'''
from typing import List


class Solution:
    def spiralOrder(self , matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
        res = []
        l,r,u,b = 0, len(matrix[0])-1,0,len(matrix)-1
        while l <= r and u <= b:
            # 从左到右
            for i in range(l,r+1):
                res.append(matrix[u][i])

            # 从上到下
            u += 1
            if u > b:
                break
            for i in range(u,b+1):
                res.append(matrix[i][r])

            # 从右到左
            r -= 1
            if r < l:
                break
            i = r
            while i >= l:
                res.append(matrix[b][i])
                i -= 1

            # 从下到上
            b -= 1
            if b < u:
                break
            i = b
            while i >= u:
                res.append(matrix[i][l])
                i -= 1

            l += 1
            if l > r:
                break
        return res
    
if __name__ == "__main__":
    print(Solution().spiralOrder([[1,2,3],[4,5,6],[7,8,9]]))
