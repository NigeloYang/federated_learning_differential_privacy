# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''基数排序'''


class Solution:
    def radix_sort(self, nums):
        m = max(nums)
        # 按照从低位到高位的顺序排序数组
        exp = 1
        while exp <= m:
            self.count_sort(nums, exp)
            exp *= 10
        return nums
    
    def count_sort(self, nums, exp):
        count = [0] * 10
        n = len(nums)
        
        # 统计 0~9 各数字的出现次数
        for i in range(n):
            d = (nums[i] // exp) % 10  # 获取 nums[i] 第 k 位的数字大小，记为 d
            count[d] += 1
        # 求前缀和
        for i in range(1, 10):
            count[i] += count[i - 1]
        
        # 倒序遍历结果
        res = [0] * n
        for i in range(n - 1, -1, -1):
            d = (nums[i] // exp) % 10
            res[count[d] - 1] = nums[i]
            count[d] -= 1
        
        # 替换结果
        for i in range(n):
            nums[i] = res[i]

if __name__ == "__main__":
    print(Solution().radix_sort([49, 99, 82, 9, 57, 43, 91, 75, 15, 37]))
