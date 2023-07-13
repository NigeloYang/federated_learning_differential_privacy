# -*- coding: utf-8 -*-
# @Time    : 2023/6/26
'''剑指 Offer 56 - I. 数组中数字出现的次数'''
from typing import List


class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        count = {}
        for num in nums:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
        res = []
        
        for k, v in count.items():
            if v == 1:
                res.append(k)
        
        return res

    def singleNumbers2(self, nums: List[int]) -> List[int]:
        x, y, n, m = 0, 0, 0, 1
        for num in nums:  # 1. 遍历异或
            n ^= num
        while n & m == 0:  # 2. 循环左移，计算 m
            m <<= 1
        for num in nums:  # 3. 遍历 nums 分组
            if num & m:
                x ^= num  # 4. 当 num & m != 0
            else:
                y ^= num  # 4. 当 num & m == 0
        return [x, y]  # 5. 返回出现一次的数字


if __name__ == "__main__":
    print(Solution().singleNumbers([4, 1, 4, 6]))
    print(Solution().singleNumbers2([4, 1, 4, 6]))
