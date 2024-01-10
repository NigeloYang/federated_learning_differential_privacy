# -*- coding: utf-8 -*-
# @Time    : 2023/10/26
from typing import List


class Solution:
    def mostFruitTree(self, fruit: List[int]) -> int:
        if not fruit:
            return 0
        
        curf = 10
        nextf = 0
        maxf = 0
        for i in range(len(fruit)):
            nextf = curf + fruit[i]
            if max(curf, nextf) > maxf:
                maxf = max(curf, nextf)
            print(maxf)
            curf = nextf
        
        return maxf
    
if __name__ == "__main__":
    print(Solution().mostFruitTree([-3,2,4,-1,-5]))
