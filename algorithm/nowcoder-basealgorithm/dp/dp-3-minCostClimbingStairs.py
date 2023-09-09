# -*- coding: utf-8 -*-
# @Time    : 2023/8/27


if __name__ == "__main__":
    n = int(input())
    cost = list(input().split(' '))
    cost = [int(i) for i in cost]
    
    dp = [0] * n
    dp[1] = min(cost[0],cost[1])
    for i in range(2, n):
        dp[i] = min(dp[i - 1] + cost[i], dp[i - 2] + cost[i - 1])
    
    print(dp[-1])
