# -*- coding: utf-8 -*-
# @Time    : 2023/10/26
from typing import List


class Solution:
    def findMissingAndMaxNegative(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        postive, negative = 1, float('-inf')
        count = set()
        for num in nums:
            if num >= 0:
                count.add(num)
            else:
                negative = max(num, negative)

        if negative == float('-inf'):
            negative = 0
        
        while postive in count:
            postive += 1
        
        return [postive, negative]
    
if __name__ == "__main__":
    print(Solution().findMissingAndMaxNegative([0,1,-2,-3,-4]))
