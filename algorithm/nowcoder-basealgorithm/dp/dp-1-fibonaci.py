# -*- coding: utf-8 -*-
# @Time    : 2023/8/26


if __name__ == "__main__":
    n = int(input())
    dp = [1, 1]
    def fibonaci(a, b):
        return a + b
    if n == 1 or n == 2:
        print('1')
    else:
        for i in range(n - 2):
            dp.append(fibonaci(dp[i], dp[i + 1]))
    print(dp[-1])