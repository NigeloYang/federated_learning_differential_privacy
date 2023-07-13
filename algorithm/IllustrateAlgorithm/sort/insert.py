# -*- coding: utf-8 -*-
# @Time    : 2023/6/21

class Solution:
    def insert_sort(self, nums):
        Leng = len(nums)
        for i in range(1, Leng):
            pivot = i - 1
            current = nums[i]
            while pivot >= 0 and current < nums[pivot]:
                nums[pivot + 1] = nums[pivot]
                pivot -= 1
            nums[pivot + 1] = current
        
        return nums


if __name__ == "__main__":
    print(Solution().insert_sort([1, 3, 4, 1, 5, 2]))
    print(Solution().insert_sort([1, 3, 4, 7, 5, 2]))
