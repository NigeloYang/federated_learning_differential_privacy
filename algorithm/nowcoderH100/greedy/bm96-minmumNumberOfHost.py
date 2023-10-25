# -*- coding: utf-8 -*-
# @Time    : 2023/9/30
from typing import List

'''贪心思想'''
class Solution:
    def minmumNumberOfHost(self, n: int, startEnd: List[List[int]]) -> int:
        if n == 0:
            return 0
        start = list()
        end = list()
        for se in startEnd:
            start.append(se[0])
            end.append(se[1])
        start.sort()
        end.sort()
        res = 0
        j = 0
        for i in range(n):
            if start[i] >= end[j]:
                j+=1
            else:
                res += 1
        return res


if __name__ == "__main__":
    print(Solution().minmumNumberOfHost(2, [[1, 2], [2, 3]]))
    print(Solution().minmumNumberOfHost(10,
                                        [[2147483646, 2147483647], [2147483646, 2147483647], [2147483646, 2147483647],
                                         [2147483646, 2147483647], [2147483646, 2147483647], [2147483646, 2147483647],
                                         [2147483646, 2147483647], [2147483646, 2147483647], [2147483646, 2147483647],
                                         [2147483646, 2147483647]]))
    print(Solution().minmumNumberOfHost(10,
                                        [[2147483646, 2147483647], [-2147483648, -2147483647], [2147483646, 2147483647],
                                         [-2147483648, -2147483647], [2147483646, 2147483647],
                                         [-2147483648, -2147483647],
                                         [2147483646, 2147483647], [-2147483648, -2147483647], [2147483646, 2147483647],
                                         [-2147483648, -2147483647]]))
