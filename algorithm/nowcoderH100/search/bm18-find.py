# -*- coding: utf-8 -*-
# @Time    : 2023/9/5

class Solution:
    def Find(self, target: int, array: List[List[int]]) -> bool:
        if len(array) == 0:
            return False
        n = len(array)
        
        if len(array[0]) == 0:
            return False
        m = len(array[0])
        
        i, j = n - 1, 0
        
        while i >= 0 and j < m:
            if array[i][j] > target:
                i -= 1
            elif array[i][j] < target:
                j += 1
            else:
                return True
        
        return False
    
if __name__ == "__main__":
    print()
