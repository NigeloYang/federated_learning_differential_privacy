# -*- coding: utf-8 -*-
# @Time    : 2023/8/27


if __name__ == "__main__":
    n = int(input())
    
    
    def skipsteps(a, b):
        return a + b
    
    
    dp = [1, 2]
    if n == 1 or n == 2:
        print(n)
    else:
        for i in range(n - 2):
            dp.append(skipsteps(dp[i], dp[i + 1]))
        print(dp[-1])
