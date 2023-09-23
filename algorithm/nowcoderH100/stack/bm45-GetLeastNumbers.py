# -*- coding: utf-8 -*-
# @Time    : 2023/9/10

class Solution:
    def GetLeastNumbers_Solution(self, input: List[int], k: int) -> List[int]:
        if len(input) < k:
            return input
        
        def quick_sort(l, r):
            i, j = l, r
            while i < j:
                if i < j and input[j] >= input[l]:
                    j -= 1
                if i < j and input[i] <= input[l]:
                    i += 1
                input[i], input[j] = input[j], input[i]
            input[l], input[i] = input[i], input[l]
            
            if i < k:
                quick_sort(l, i)
            if i > k:
                quick_sort(i + 1, r)
            
            return input[:k]
        
        return quick_sort(0, len(input) - 1)

if __name__ == "__main__":
    print()
