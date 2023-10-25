# -*- coding: utf-8 -*-
# @Time    : 2023/10/24
from typing import List


class Solution:
    def findFeedOrderII(self, numCows: int, feedOrders: List[List[int]]) -> List[int]:
        nums = [[] for i in range(numCows)]
        indegrees = [0] * numCows
        zeroinde = []
        res = []
        
        for feedOrder in feedOrders:
            nums[feedOrder[1]].append(feedOrder[0])
            indegrees[feedOrder[0]] += 1
        for i in range(numCows):
            if indegrees[i] == 0:
                zeroinde.append(i)
        
        while zeroinde:
            cowi = zeroinde.pop()
            numCows -= 1
            res.append(cowi)
            for num in nums[cowi]:
                indegrees[num] -= 1
                if indegrees[num] == 0:
                    zeroinde.append(num)
        
        if numCows == 0:
            return res
        else:
            return []


if __name__ == "__main__":
    print(Solution().findFeedOrderII(2, [[1, 0]]))
    print(Solution().findFeedOrderII(4, [[1, 0], [2, 0], [3, 1], [3, 2], [1, 2]]))
