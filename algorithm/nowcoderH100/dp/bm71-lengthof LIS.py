# -*- coding: utf-8 -*-
# @Time    : 2023/9/16

class Solution:
    def LIS(self, arr: List[int]) -> int:
        n = len(arr)
        
        dp = [1] * n
        
        for i in range(n):
            for j in range(i):
                if arr[j] < arr[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)

if __name__ == "__main__":
    print()
