# -*- coding: utf-8 -*-
# @Time    : 2023/6/21

''' 快速排序思想
1.首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置。
2.再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。
3.重复第二步，直到所有元素均排序完毕。
'''
class Solution:
    def selected_sort(self, nums):
        for i in range(len(nums) - 1):
            mini = i
            for j in range(i + 1, len(nums)):
                if nums[j] < nums[mini]:
                    mini = j
            if i != mini:
                nums[i], nums[mini] = nums[mini], nums[i]
        return nums


if __name__ == "__main__":
    print(Solution().selected_sort([1, 3, 4, 1, 5, 2]))
    print(Solution().selected_sort([1, 3, 4, 7, 5, 2]))
