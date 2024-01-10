# -*- coding: utf-8 -*-
# @Time    : 2023/10/25
from typing import List


class Solution:
    def findGatheringAreas(self , groups: List[int], n: int) -> List[List[int]]:
        if not groups:
            return []
        s,f = 0,1
        ans = []
        while f < len(groups):
            if groups[f] - groups[f-1] == 1:
                f += 1
            else:
                ans.append([groups[s],groups[f-1]])
                s = f
                f += 1

        ans.append([groups[s],groups[f-1]])
        return ans
    
if __name__ == "__main__":
    print(Solution().findGatheringAreas([-2, -1, 0, 3, 4, 6],6))
