# -*- coding: utf-8 -*-
# @Time    : 2023/7/4

''' 滑动窗口最大值
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。'''
import collections


class Solution:
    def maxSlidingWindow(self, nums, k: int):
        # 超出时间限制
        if k == 1:
            return nums
        res = []
        for i in range(k, len(nums) + 1):
            temp = max(nums[i - k:i])
            res.append(temp)
        return res
    
    def maxSlidingWindow2(self, nums, k: int):
        #　超出时间限制
        if k == 1:
            return nums
        n = len(nums)
        res = []
        queue = []
        for i in range(n):
            # 如果当前队列最左侧存储的下标等于 i - k  的值，代表目前队列已满。
            # 但是新元素需要进来，所以列表最左侧的下标出队列
            if queue and queue[0] == i - k:
                queue.pop(0)
                # queue.popleft()
            # 对于新进入的元素，如果队列前面的数比它小，那么前面的都出队列
            while queue and nums[queue[-1]] < nums[i]:
                queue.pop()
            queue.append(i)
            # 当前的大值加入到结果数组中
            if i >= k - 1:
                res.append(nums[queue[0]])
        return res
    def maxSlidingWindow3(self, nums, k: int):
        if k == 1:
            return nums
        res = []
        queue = collections.deque()
        for i in range(len(nums)):
            # 如果当前队列最左侧存储的下标等于 i - k  的值，代表目前队列已满。
            # 但是新元素需要进来，所以列表最左侧的下标出队列
            if queue and queue[0] == i - k:
                queue.popleft()
            # 对于新进入的元素，如果队列前面的数比它小，那么前面的都出队列
            while queue and nums[queue[-1]] < nums[i]:
                queue.pop()
            queue.append(i)
            # 当前的大值加入到结果数组中
            if i >= k - 1:
                res.append(nums[queue[0]])
        return res


if __name__ == "__main__":
    print(Solution().maxSlidingWindow([1], 1))
    print(Solution().maxSlidingWindow([1, -1], 1))
    print(Solution().maxSlidingWindow2([1, -1], 1))
    print(Solution().maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3))
    print(Solution().maxSlidingWindow2([1, 3, -1, -3, 5, 3, 6, 7], 3))
    print(Solution().maxSlidingWindow2([1, 3, 1, 2, 0, 5], 3))
