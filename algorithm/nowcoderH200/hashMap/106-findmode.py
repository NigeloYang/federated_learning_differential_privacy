# -*- coding: utf-8 -*-
# @Time    : 2023/11/1
from typing import List


class Solution:
    def findMode(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        dic = {}
        ans = []
        maxcount = 0
        maxnum = 0
        
        for i in range(len(nums)):
            if nums[i] in dic:
                dic[nums[i]] += 1
            else:
                dic[nums[i]] = 1
            if dic[nums[i]] >= maxcount:
                maxnum =  max(maxnum,nums[i]) if dic[nums[i]] == maxcount else nums[i]
                maxcount = dic[nums[i]]
            
            ans.append(maxnum)
        
        return ans

if __name__ == "__main__":
    print(Solution().findMode([1, 2, 3, 2]))
