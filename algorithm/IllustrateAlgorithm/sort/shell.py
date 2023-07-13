# -*- coding: utf-8 -*-
# @Time    : 2023/6/22

''' 希尔排序
选择一个增量序列 t1，t2，……，tk，其中 ti > tj, tk = 1；
按增量序列个数 k，对序列进行 k 趟排序；
每趟排序，根据对应的增量 ti，将待排序列分割成若干长度为 m 的子序列，分别对各子表进行直接插入排序。
仅增量因子为 1 时，整个序列作为一个表来处理，表长度即为整个序列的长度
'''


class Solution:
    def sheel_sort(self, nums):
        gap = 1
        while (gap < len(nums) // 3):
            gap = gap * 3 + 1
        while gap > 0:
            # gap=1完整的插入排序，相反则是根据增量进行插入排序
            for i in range(gap, len(nums)):
                current = nums[i]
                j = i - gap
                flag = True
                while j >= 0 and nums[j] > current:
                    nums[j + gap] = nums[j]
                    j -= gap
                    flag = True
                if flag:
                    nums[j + gap] = current
            gap = gap // 3

        return nums


if __name__ == "__main__":
    print(Solution().sheel_sort([4, 3, 2, 7, 5, 2, 11, 10]))
