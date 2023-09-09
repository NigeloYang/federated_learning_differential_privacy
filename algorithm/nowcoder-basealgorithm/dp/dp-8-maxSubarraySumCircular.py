# -*- coding: utf-8 -*-
# @Time    : 2023/8/29


if __name__ == "__main__":
    n = int(input())
    nums = list(map(int, input().split()))
    
    dp1 = [0] * n
    dp2 = [0] * n
    dp1[0] = dp2[0] = nums[0]
    
    for i in range(1, n):
        dp1[i] = min(nums[i] + dp1[i - 1], nums[i])
        dp2[i] = max(nums[i] + dp2[i - 1], nums[i])
    
    t1 = min(dp1)
    if t1 < 0:
        t1 = sum(nums) - t1
    
    t2 = max(dp2)
    if t2 < 0:
        print(t2)
    else:
        print(max(max(dp2), t1))
