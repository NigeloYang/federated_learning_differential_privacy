# -*- coding: utf-8 -*-
# @Time    : 2023/9/10

class Solution:
    def findKth(self, a: List[int], n: int, K: int) -> int:
        res = self.quicksort(a, 0, n-1)

        return res[-K]

    def quicksort(self, arr, l, r):
        # 子数组长度为 1 时终止递归
        if l >= r: return
        # 哨兵划分操作（以 arr[l] 作为基准数）
        i, j = l, r
        while i < j:
            while i < j and arr[j] >= arr[l]:
                j -= 1
            while i < j and arr[i] <= arr[l]:
                i += 1
            arr[i], arr[j] = arr[j], arr[i]
        arr[l], arr[i] = arr[i], arr[l]

        self.quicksort(arr, l, i - 1)
        self.quicksort(arr, i + 1, r)

        return arr
    
if __name__ == "__main__":
    print()
