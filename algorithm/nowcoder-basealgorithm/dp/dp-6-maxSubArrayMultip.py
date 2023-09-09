# -*- coding: utf-8 -*-
# @Time    : 2023/8/27


if __name__ == "__main__":
    n = int(input())
    nums = list(map(int, input().split()))
    res = nums[0]
    minnum, maxnum = nums[0], nums[0]
    
    for cur in nums[1:]:
        te1, te2 = minnum * cur, maxnum * cur
        maxnum = max(te1, te2, cur)
        minnum = min(te1, te2, cur)
        if res < maxnum:
            res = maxnum
    print(res)

