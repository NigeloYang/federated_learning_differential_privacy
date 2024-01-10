# -*- coding: utf-8 -*-
# @Time    : 2023/10/30

'''
在一个m x n的牧场中，遭受了疯牛病的袭击，有部分的牛感染了病毒。

现在，牧场中每个单元格可以有以下三个值之一：

值0代表空地；
值1代表健康的牛；
值2代表患有疯牛病的牛。
每分钟，患有疯牛病的牛周围4个方向上相邻的健康牛都会被感染。

返回直到牧场中没有健康的牛为止所必须经过的最小分钟数。如果不可能到达此情况，返回-1。
'''
from typing import List
import collections

class Solution:
    def healthyCowsII(self, pasture: List[List[int]]) -> int:
        if not pasture:
            return -1
        ans = 0
        count = 0
        badhealthy = collections.deque()
        for i in range(len(pasture)):
            for j in range(len(pasture[0])):
                if pasture[i][j] == 2:
                    badhealthy.append((i,j))
                elif pasture[i][j] ==1:
                    ans += 1
        while badhealthy:
            for _ in range(len(badhealthy)):
                c_i, c_j = badhealthy.popleft()
                for i, j in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
                    n_i, n_j = c_i + i, c_j + j
                    if 0 <= n_i < len(pasture) and 0 <= n_j < len(pasture[0]) and pasture[n_i][n_j] == 1:
                        pasture[n_i][n_j] = 2
                        ans -= 1
                        badhealthy.append((n_i, n_j))
            count += 1
        return -1 if ans > 0 else count - 1
        

if __name__ == "__main__":
    print(Solution().healthyCowsII([[2, 1, 1],
                                    [0, 1, 1],
                                    [1, 0, 1]]))
    print(Solution().healthyCowsII([[1, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 2]]))
    print(Solution().healthyCowsII([[2,1,1],[0,1,1],[1,0,1]]))
