# -*- coding: utf-8 -*-
# @Time    : 2023/9/14

class Solution:
    def LCS(self, str1: str, str2: str) -> str:
        l1, l2 = len(str1), len(str2)
        
        dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]
        max = 0
        pos = 0
        
        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                
                if dp[i][j] > max:
                    max = dp[i][j]
                    pos = i - 1
        return str1[pos - max + 1:pos + 1]

if __name__ == "__main__":
    print()
