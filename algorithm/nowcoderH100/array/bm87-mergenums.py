# -*- coding: utf-8 -*-
# @Time    : 2023/9/27
from typing import List


class Solution:
    def merge(self, A, m, B, n):
        if m == 0 and m == 0:
            return []
        
        res = []
        i, j = 0, 0
        while i < m and j < n:
            if A[i] < B[j]:
                res.append(A[i])
                i += 1
            else:
                res.append(B[j])
                j += 1
        if i < m and j == n:
            res.append(A[i:])
        elif i == m and j < n:
            temp = B[j:]
            res.extend(temp)
        return res

    def merge2(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        p1, p2, p = m - 1, n - 1, m + n - 1
        while p2 >= 0:  # nums2 还有要合并的元素
            # 如果 p1 < 0，那么走 else 分支，把 nums2 合并到 nums1 中
            if p1 >= 0 and nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]  # 填入 nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]  # 填入 nums2[p1]
                p2 -= 1
            p -= 1  # 下一个要填入的位置

if __name__ == "__main__":
    print(Solution().merge([4,5,6],3,[1,2,3],3))
    print(Solution().merge2([4,5,6],3,[1,2,3],3))
