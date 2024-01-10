# -*- coding: utf-8 -*-
# @Time    : 2023/10/27

'''
牧场主有一个产奶数据的整数数组 milk_amount，表示不同品种的乳牛的产奶量。请返回一个数组 others表示其他品种的牛产奶量的乘积，其中 others[i] 等于数组 milk_amount 中除了 milk_amount[i] 之外其他元素的乘积。
题目数据保证数组 milk_amount 中任意元素的全部品种索引前元素和后元素的产奶量乘积都在 32 位整数范围内。
'''
from typing import List


class Solution:
    def product_except_self(self, milks: List[int]) -> List[int]:
        if not milks:
            return []
        
        count1 = 1
        ans = [1] * len(milks)
        for i in range(1, len(milks)):
            count1 *= milks[i - 1]
            ans[i] = count1
        
        count2 = 1
        for i in range(len(milks) - 2, -1, -1):
            count2 *= milks[i + 1]
            ans[i] *= count2
        
        return ans



if __name__ == "__main__":
    print(Solution().product_except_self([5, 7, 3, 1]))
