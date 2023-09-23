# -*- coding: utf-8 -*-
# @Time    : 2023/9/14


class Solution:
    def LCS(self, s1: str, s2: str) -> str:
        if len(s1) == 0 or len(s2) == 0:
            return "-1"
        l1, l2 = len(s1), len(s2)
        
        dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]
        
        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        
        i, j = l1, l2
        s = []
        while dp[i][j] != 0:
            if dp[i][j] == dp[i][j - 1]:
                j -= 1
            elif dp[i][j] == dp[i - 1][j]:
                i -= 1
            elif dp[i][j] >= dp[i - 1][j - 1]:
                i -= 1
                j -= 1
                s.append(s1[i])
        if len(s) == 0:
            return '-1'
        else:
            return ''.join(s[::-1])
        
if __name__ == "__main__":
    print(Solution().LCS("1A2C3D4B56","B1D23A456A"))
