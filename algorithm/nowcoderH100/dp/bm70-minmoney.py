# -*- coding: utf-8 -*-
# @Time    : 2023/9/16

class Solution:
    def minMoney(self, arr: List[int], aim: int) -> int:
        n = len(arr)
        dp = [[aim + 1] * (aim + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(aim + 1):
                for k in range(j // arr[i - 1] + 1):
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - k * arr[i - 1]] + k)
        
        ans = dp[n][aim]
        
        return ans if ans != aim + 1 else -1
    def minMoney2(self , arr: List[int], aim: int) -> int:
        n = len(arr)
        dp = [[aim+1] * (aim+1) for _ in range(n+1)]
        dp[0][0] = 0

        for i in range(1,n+1):
            for j in range(aim+1):
                if j < arr[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-arr[i-1]] + 1)

        ans = dp[n][aim]

        return ans if ans != aim+1 else -1

if __name__ == "__main__":
    print()
