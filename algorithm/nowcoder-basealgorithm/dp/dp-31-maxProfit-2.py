# -*- coding: utf-8 -*-
# @Time    : 2023/8/28


if __name__ == "__main__":
    n = int(input())
    prices = list(map(int, input().split()))
    profit = 0
    
    for i in range(1, n):
        if prices[i] - prices[i - 1] > 0:
            profit += prices[i] - prices[i - 1]
    
    print(profit)
