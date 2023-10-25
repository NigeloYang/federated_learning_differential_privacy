# -*- coding: utf-8 -*-
# @Time    : 2023/9/28
from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        res = []
        for interval in intervals:
            if not res or res[-1][-1] < interval[0]:
                res.append(interval)
            else:
                res[-1][-1] = max(res[-1][-1], interval[-1])
        return res
if __name__ == "__main__":
    print()
