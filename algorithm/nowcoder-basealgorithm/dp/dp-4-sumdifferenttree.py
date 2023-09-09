# -*- coding: utf-8 -*-
# @Time    : 2023/8/27


if __name__ == "__main__":
    n = int(input())
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1
    
    for i in range(2, n + 1):
        for j in range(1, i + 1):
            dp[i] += dp[j - 1] * dp[i - j]
    
    print(dp[-1])