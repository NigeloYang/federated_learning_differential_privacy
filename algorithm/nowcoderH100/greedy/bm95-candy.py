# -*- coding: utf-8 -*-
# @Time    : 2023/9/30
from typing import List


class Solution:
    def candy(self, ratings: List[int]) -> int:
        if not ratings:
            return 0
        n = len(ratings)
        left = [1 for i in range(n)]
        right = left[:]
        for i in range(1,n):
            if ratings[i] > ratings[i-1]:
                left[i] = left[i-1] +1
        count = left[-1]
        for i in range(n-2,-1,-1):
            if ratings[i] > ratings[i+1]:
                right[i] = right[i+1] + 1
            count += max(left[i],right[i])
        return count

if __name__ == "__main__":
    print(Solution().candy([1,1,2]))
