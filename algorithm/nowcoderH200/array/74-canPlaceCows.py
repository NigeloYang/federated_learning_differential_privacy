# -*- coding: utf-8 -*-
# @Time    : 2023/10/26
from typing import List


class Solution:
    def canPlaceCows(self, pasture: List[int], n: int) -> bool:
        if not pasture:
            return False
        for i in range(len(pasture) - 1):
            if pasture[i] != 1:
                if pasture[i - 1] != 1 and pasture[i + 1] != 1:
                    n -= 1
                elif i == 0 and pasture[i + 1] == 0:
                    n -= 1
        if pasture[len(pasture) - 1] == 0 and pasture[len(pasture) - 2] == 0:
            n -= 1
        return n <= 0


if __name__ == "__main__":
    print(Solution().canPlaceCows([1, 0, 0, 0, 1], 1))
