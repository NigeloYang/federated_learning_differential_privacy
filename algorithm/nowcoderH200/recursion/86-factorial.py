# -*- coding: utf-8 -*-
# @Time    : 2023/10/27

class Solution:
    def factorial(self, n: int) -> int:
        ans = 1
        for i in range(2,n+1):
            ans *= i
        return ans % 1000000007
        
            
    

if __name__ == "__main__":
    print(Solution().factorial(3))
    print(Solution().factorial(4))
