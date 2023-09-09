# -*- coding: utf-8 -*-
# @Time    : 2023/8/28
import sys

if __name__ == "__main__":

    n = int(sys.stdin.readline().strip())
    prices = list(map(int, sys.stdin.readline().strip().split()))
    
    minp = int(1e9)
    maxp = 0
    
    for price in prices:
        minp = min(minp, price)
        maxp = max(maxp, price - minp)
    print(maxp)
