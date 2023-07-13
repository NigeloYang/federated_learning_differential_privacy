# -*- coding: utf-8 -*-
# @Time    : 2023/6/22

'''剑指 Offer 40. 最小的 k 个数
输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
'''
from typing import List


class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k >= len(arr):
            return arr
        
        def quick_sort(arr, start, end):
            # 哨兵分区
            if start >= end:
                return
            i, j = start, end
            while i < j:
                while i < j and arr[j] >= arr[start]:
                    j -= 1
                while i < j and arr[i] <= arr[start]:
                    i += 1
                arr[i], arr[j] = arr[j], arr[i]
            arr[start], arr[i] = arr[i], arr[start]
            # 递归快速排序
            quick_sort(arr, start, i - 1)
            quick_sort(arr, i + 1, end)
        
        quick_sort(arr, 0, len(arr) - 1)
        return arr[:k], arr
    
    def getLeastNumbers2(self, arr: List[int], k: int) -> List[int]:
        if k >= len(arr):
            return arr
        
        def quick_sort(start, end):
            # 哨兵分区
            if start >= end:
                return
            i, j = start, end
            while i < j:
                while i < j and arr[j] >= arr[start]:
                    j -= 1
                while i < j and arr[i] <= arr[start]:
                    i += 1
                arr[i], arr[j] = arr[j], arr[i]
            arr[start], arr[i] = arr[i], arr[start]
            # 递归快速排序
            if k < i:
                return quick_sort(start, i - 1)
            if k > i:
                return quick_sort(i + 1, end)
            return arr[:k]
        
        return quick_sort(0, len(arr) - 1)


if __name__ == "__main__":
    data2 = [2, 4, 1, 0, 3, 5]
    print(Solution().getLeastNumbers(data2, 2))
    print(Solution().getLeastNumbers2(data2, 2))
