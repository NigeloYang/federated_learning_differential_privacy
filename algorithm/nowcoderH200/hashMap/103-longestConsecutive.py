# -*- coding: utf-8 -*-
# @Time    : 2023/10/31
from typing import List


class Solution:
    def longestConsecutive(self, tag: List[int]) -> int:
        if not tag:
            return
        record = set()
        for v in tag:
            if v not in record:
                record.add(v)
        print(record)
        maxlen = 0
        for v in tag:
            if v - 1 in record:
                continue
            curv = v
            curlen = 1
            while curv + 1 in record:
                curv += 1
                curlen += 1
            maxlen = max(maxlen, curlen)
        
        return maxlen


if __name__ == "__main__":
    print(Solution().longestConsecutive([10, 4, 20, 1, 3, 2]))
