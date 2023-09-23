# -*- coding: utf-8 -*-
# @Time    : 2023/9/16


class Solution:
    def minNumberDisappeared(self, nums: List[int]) -> int:
        res = 1
        dic = dict()
        
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
        
        for i in range(len(nums)):
            if res not in dic:
                return res
            else:
                res += 1
        
        return res
    
if __name__ == "__main__":
    print()
