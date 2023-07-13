# -*- coding: utf-8 -*-
# @Time    : 2023/6/21

class Solution:
    def bubble_sort(self, nums):
        N = len(nums)
        for i in range(N - 1):  # 外循环
            for j in range(N - i - 1):  # 内循环
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        return nums
    
    # 优化
    def bubble_sort2(self, nums):
        N = len(nums)
        for i in range(N - 1):  # 外循环
            flag = False
            for j in range(N - i - 1):  # 内循环
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
                    flag = True
            if not flag:
                break
        return nums


if __name__ == "__main__":
    print(Solution().bubble_sort([1, 3, 4, 1, 5, 2]))
    print(Solution().bubble_sort2([1, 2, 3, 4, 5]))
