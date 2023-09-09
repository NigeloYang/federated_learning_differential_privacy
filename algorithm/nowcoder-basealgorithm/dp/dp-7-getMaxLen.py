# -*- coding: utf-8 -*-
# @Time    : 2023/8/27


if __name__ == "__main__":
    # 超时
    n = int(input())
    nums = list(map(int, input().split()))
    
    count = 0
    minnum, maxnum = 1, 1
    
    for i in range(n):
        t1, t2 = minnum * nums[i], maxnum * nums[i]
        maxnum = max(t1, t2)
        minnum = min(t1, t2)
        
        if maxnum > 0:
            count = i
    print(count + 1)
    
    # 2
    pos, neg = [0] * n, [0] * n
    pos[0] = int(nums[0] > 0)
    neg[0] = int(nums[0] < 0)
    for i in range(1, n):
        if nums[i] == 0:
            pos[i], neg[i] = 0, 0
        elif nums[i] > 0:
            pos[i] = pos[i - 1] + 1
            neg[i] = neg[i - 1] + 1 if neg[i - 1] > 0 else 0
        elif nums[i] < 0:
            pos[i] = neg[i - 1] + 1 if neg[i - 1] > 0 else 0
            neg[i] = pos[i - 1] + 1
    print(max(pos))
    
    #  3
    pre = -1
    stack = []
    res = 0
    for i, num in enumerate(nums):
        if num < 0:
            stack.append(i)
        elif num == 0:
            stack, pre = [], i
        if len(stack) % 2 == 0:
            res = max(res, i - pre)
        else:
            res = max(res, i - stack[0])
    print(res)
