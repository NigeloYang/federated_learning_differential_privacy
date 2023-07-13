# -*- coding: utf-8 -*-
# @Time    : 2023/6/26


'''剑指 Offer 56 - II. 数组中数字出现的次数 II

在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。
'''
from typing import List


class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        count = {}
        for num in nums:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
        
        for k, v in count.items():
            if v == 1:
                return k
    
    def singleNumbers2(self, nums: List[int]) -> List[int]:
        one, two = 0, 0
        for num in nums:
            one = one ^ num & ~two
            two = two ^ num & ~one
        return one


    def singleNumbers3(self, nums: List[int]) -> int:
        counts = [0] * 32
        for num in nums:
            for i in range(32):
                counts[i] += num & 1  # 更新第 i 位 1 的个数之和
                num >>= 1  # 第 i 位 --> 第 i 位
        res, m = 0, 3
        for i in range(31, -1, -1):
            res <<= 1
            res |= counts[i] % m  # 恢复第 i 位
        return res if counts[31] % m == 0 else ~(res ^ 0xffffffff)


if __name__ == "__main__":
    print(Solution().singleNumbers([3, 4, 3, 3]))
    print(Solution().singleNumbers2([3, 4, 3, 3]))
    print(Solution().singleNumbers3([3, 4, 3, 3]))
