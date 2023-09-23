# -*- coding: utf-8 -*-
# @Time    : 2023/9/16

class Solution:
    def FindNumsAppearOnce(self, nums: List[int]) -> List[int]:
        dic = dict()
        
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
        
        res = []
        for k, v in dic.items():
            if v == 1:
                res.append(k)
        res.sort()
        return res

if __name__ == "__main__":
    print()
