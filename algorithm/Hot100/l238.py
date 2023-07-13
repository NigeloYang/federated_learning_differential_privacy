# -*- coding: utf-8 -*-
# @Time    : 2023/7/5

''' 除自身以外数组的乘积
给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。

题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。

请不要使用除法，且在 O(n) 时间复杂度内完成此题。'''


class Solution:
    def productExceptSelf(self, nums):
        # 超时
        res = []
        for i in range(len(nums)):
            temp = 1
            for j in range(len(nums)):
                if i != j:
                    temp *= nums[j]
            res.append(temp)
        return res
    
    def productExceptSelf2(self, nums):
        n = len(nums)
        res = [1]
        l, r = 1, 1
        for i in range(n - 1):
            r *= nums[i]
            res.append(r)
        for i in range(n - 1, 0, -1):
            l *= nums[i]
            res[i - 1] *= l
        return res


if __name__ == "__main__":
    print(Solution().productExceptSelf([1, 2, 3, 4]))
    print(Solution().productExceptSelf2([1, 2, 3, 4]))
    

