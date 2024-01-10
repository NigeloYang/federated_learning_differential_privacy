# -*- coding: utf-8 -*-
# @Time    : 2023/10/30
from typing import List
import collections


class Solution:
    def healthyCows(self, pasture: List[List[int]], k: int) -> int:
        ans = 0
        badhealthy = collections.deque()
        for i in range(len(pasture)):
            for j in range(len(pasture[0])):
                if pasture[i][j] == 2:
                    badhealthy.append((i, j))
                elif pasture[i][j] == 1:
                    ans += 1
        while badhealthy and k > 0:
            for _ in range(len(badhealthy)):
                c_i, c_j = badhealthy.popleft()
                for i, j in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
                    n_i, n_j = c_i + i, c_j + j
                    if 0 <= n_i < len(pasture) and 0 <= n_j < len(pasture[0]) and pasture[n_i][n_j] == 1:
                        pasture[n_i][n_j] = 2
                        ans -= 1
                        badhealthy.append((n_i, n_j))
            k -= 1
        return ans if ans > 0 else 0


if __name__ == "__main__":
    print(Solution().healthyCows([[2, 1, 1], [1, 1, 0], [0, 1, 1]], 2))
    print(Solution().healthyCows([[1, 1, 1, 1, 1, 1, 1, 1, 1, 2]], 4))
