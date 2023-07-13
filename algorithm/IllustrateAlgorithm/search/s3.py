# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''剑指 Offer 11. 旋转数组的最小数字
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。

给你一个可能存在 重复 元素值的数组 numbers ，它原来是一个升序排列的数组，并按上述情形进行了一次旋转。
请返回旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一次旋转，该数组的最小值为 1。  

注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
'''
from typing import List


class Solution:
    def minArray(self, numbers: List[int]) -> int:
        if len(numbers) <= 1:
            return numbers[0]

        mini = numbers[0]
        for number in numbers[1:]:
            if number < mini:
                return number
        return mini
    
if __name__ == "__main__":
    print(Solution().minArray([3,4,5,1,2]))
