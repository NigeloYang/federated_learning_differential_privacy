# -*- coding: utf-8 -*-
# @Time    : 2023/8/27


if __name__ == "__main__":
    n = int(input())
    arr = list(map(int, input().split()))
    print(arr)
    
    dp = [0] * n
    for i in range(n):
        if i == 0:
            dp[i] = arr[i]
        else:
            dp[i] = max(arr[i], arr[i] + dp[i - 1])
    print(max(dp))
    
    # 2
    def maxSubArray(nums) -> int:
        for i in range(1, len(nums)):
            nums[i] += max(nums[i - 1], 0)
        return max(nums)
