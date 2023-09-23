# -*- coding: utf-8 -*-
# @Time    : 2023/9/14

class Solution:
    def uniquePaths(self , m: int, n: int) -> int:
        dp = [1] * n

        for i in range(1,n):
            for j in range(1,m):
                dp[j] = dp[j] + dp[j-1]
        return dp[-1]
    
if __name__ == "__main__":
    print()
