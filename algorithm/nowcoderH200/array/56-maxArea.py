# -*- coding: utf-8 -*-
# @Time    : 2023/10/23
from typing import List


class Solution:
    def maxArea(self, areas: List[int]) -> int:
        if not areas:
            return
        
        areas.sort()
        ans = []
        for i, v in enumerate(areas):
            ans.append(v * (len(areas) - i))
        return max(ans)
    
if __name__ == "__main__":
    print()
