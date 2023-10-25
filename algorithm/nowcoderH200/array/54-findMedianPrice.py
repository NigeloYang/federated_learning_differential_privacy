# -*- coding: utf-8 -*-
# @Time    : 2023/10/23
from typing import List


class Solution:
    def findMedianPrice(self, prices: List[int]) -> List[float]:
        if not prices:
            return []
        ans = []
        for i in range(len(prices)):
            temp = self.getmedian(prices[:i+1])
            ans.append(temp)
        return ans
    
    def getmedian(self,nums):
        n = len(nums)
        nums.sort()
        if n % 2 == 1:
            return nums[n//2]
        else:
            return (nums[n//2] + nums[n//2 - 1])/2
        
    def findMedianPrice2(self, prices: List[int]) -> List[float]:
        if not prices:
            return []
        ans = []
        for i in range(len(prices)):
            nums = prices[:i+1]
            n = len(nums)
            nums.sort()
            if n % 2 == 1:
                ans.append(nums[n // 2])
            else:
                ans.append((nums[n // 2] + nums[n // 2 - 1]) / 2)
        return ans


if __name__ == "__main__":
    print(Solution().findMedianPrice([100, 200, 300]))
    print(Solution().findMedianPrice2([100, 200, 300]))
    print(Solution().findMedianPrice([1, 5, 2, 4, 3]))
    print(Solution().findMedianPrice2([1, 5, 2, 4, 3]))
    print(3//2)
    print(3/2)
