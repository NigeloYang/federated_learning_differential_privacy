# -*- coding: utf-8 -*-
# @Time    : 2023/6/21

'''核心思想：哨兵划分（用于划分数据大小）和递归思想（根据哨兵的划分不停的进行递归直到完成排序）'''
class Solution:
    def quick_sort(self, nums, l, r):
        if l >= r:
            return
        i = self.partition(nums, l, r)
        self.quick_sort(nums, l, i - 1)
        self.quick_sort(nums, i + 1, r)
        
        return nums
    
    def partition(self, nums, l, r):
        i, j = l, r
        while i < j:
            while i < j and nums[j] >= nums[l]:
                j -= 1
            while i < j and nums[i] <= nums[l]:
                i += 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[l], nums[i] = nums[i], nums[l]
        
        return i


if __name__ == "__main__":
    data1 = [1, 3, 4, 1, 5, 2]
    data2 = [1, 3, 4, 7, 5, 2]
    print(Solution().quick_sort(data1, 0, len(data1) - 1))
    print(Solution().quick_sort(data2, 0, len(data2) - 1))
