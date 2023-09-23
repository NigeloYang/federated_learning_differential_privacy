# -*- coding: utf-8 -*-
# @Time    : 2023/9/16


class Solution:
    def FindGreatestSumOfSubArray(self , array: List[int]) -> int:
        if not array:
            return 0
        n = len(array)
        dp = [0] * (n)
        dp[0] = array[0]
        res = array[0]

        for i in range(1,n):
            dp[i] = max(dp[i-1]+array[i],array[i])
            res = max(dp[i],res)
        return res
    
if __name__ == "__main__":
    print()
