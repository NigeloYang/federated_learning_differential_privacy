# -*- coding: utf-8 -*-
# @Time    : 2023/10/24
from typing import List


class Solution:
    def canFeedAllCows(self , numCows: int, feedOrders: List[List[int]]) -> bool:
        nums = [[] for i in range(numCows)]
        indegrees = [0] * numCows
        zerop = []
        for feedOrder in feedOrders:
            a = feedOrder[0]
            b = feedOrder[1]
            nums[b].append(a)
            indegrees[a] += 1
        
        for i in range(numCows):
            if indegrees[i] == 0:
                zerop.append(i)
        
        while zerop:
            cowi = zerop.pop(0)
            numCows -= 1
            for i in nums[cowi]:
                indegrees[i] -= 1
                if indegrees[i] == 0:
                    zerop.append(i)
        if numCows:
            return False
        else:
            return True
if __name__ == "__main__":
    print(Solution().canFeedAllCows(2,[[1,0],[0,1]]))
