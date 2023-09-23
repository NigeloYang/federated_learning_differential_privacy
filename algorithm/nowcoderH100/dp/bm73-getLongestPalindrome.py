# -*- coding: utf-8 -*-
# @Time    : 2023/9/16

class Solution:
    def getLongestPalindrome(self, A: str) -> int:
        n = len(A)
        if n < 2:
            return n
        
        b, maxl = 0, 1
        dp = [[False] * n for i in range(n)]
        
        for i in range(n):
            dp[i][i] = True
        
        for l in range(2, n + 1):
            for i in range(n):
                r = l + i - 1
                if r >= n:
                    break
                
                if A[i] == A[r]:
                    if r - i < 3:
                        dp[i][r] = True
                    else:
                        dp[i][r] = dp[i + 1][r - 1]
                if dp[i][r] and r - i + 1 > maxl:
                    maxl = r - i + 1
                    b = i
        return len(A[b:b + maxl])

if __name__ == "__main__":
    print()
