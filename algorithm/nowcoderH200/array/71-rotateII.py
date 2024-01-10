# -*- coding: utf-8 -*-
# @Time    : 2023/10/25

'''
在一个农场里，有n头牛，每头牛都有一个唯一的编号，编号从1到n。牛群被安排在一个n的方阵中，方阵的左上角是编号为1的牛，方阵的右下角是编号为n*n的牛。
农场的主人想改变牛群的排列顺序，他的规则是：每次将方阵中的每头牛同时向左一格和向上一格移动，如果已经在最左边的牛会移动到同一行的最右边，同时向上一格，如果已经在最上边的牛会移动到同一列的最下边，同时向左一格。当最左上的牛移动时，会循环到最右下。

现在农场主人想知道，经过k次移动后，方阵中的牛群的排列情况。由于牛群数量较多，农场主人难以计算，他需要你的帮忙。'''
from typing import List


class Solution:
    def rotateII(self, n: int, k: int) -> List[List[int]]:
        if n == 0:
            return []
        nums = [[] for i in range(n)]
        for i in range(n):
            for j in range(n):
                nums[i].append(n * i + j + 1)
        k %= n
        if k == 0:
            return nums
        else:
            for i in range(n):
                l1 = nums[i][:k]
                r1 = nums[i][k:]
                r1.extend(l1)
                nums[i][:] = r1
        
        nums[:-k], nums[-k:] = nums[k:], nums[:k]
        # nums[:k], nums[k:] = nums[-k:], nums[:-k]
        
        return nums
    
    def rotateII2(self, n: int, k: int) -> List[List[int]]:
        if n == 0:
            return []
        
        ans = [[0] * n for i in range(n)]
        k %= n
        for i in range(n):
            for j in range(n):
                ni = (i + k) % n
                nj = (j + k) % n
                ans[i][j] += n * ni + nj + 1
        return ans


if __name__ == "__main__":
    print(Solution().rotateII(3, 1))
    print(Solution().rotateII2(3, 1))
