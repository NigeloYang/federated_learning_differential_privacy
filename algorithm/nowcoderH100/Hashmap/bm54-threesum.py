# -*- coding: utf-8 -*-
# @Time    : 2023/9/16

class Solution:
    def threeSum(self, num: List[int]) -> List[List[int]]:
        res = []
        n = len(num)
        if n < 3:
            return res
        
        num.sort()
        for i in range(n - 2):
            if i != 0 and num[i] == num[i - 1]:
                continue
            
            l = i + 1
            r = n - 1
            
            target = -num[i]
            
            while l < r:
                if num[l] + num[r] == target:
                    res.append([num[i], num[l], num[r]])
                    while l + 1 < r and num[l] == num[l + 1]:
                        l += 1
                    while r - 1 > l and num[r] == num[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
                elif num[l] + num[r] > target:
                    r -= 1
                else:
                    l += 1
        return res
    
if __name__ == "__main__":
    print()
