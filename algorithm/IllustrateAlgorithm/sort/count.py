# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''计数排序'''


class Solution:
    def counting_sort(self, nums):
        # 求出最大值
        ma = max(nums)
        
        # 生成数组和统计每个数字出现的次数
        count = [0] * (ma + 1)
        for num in nums:
            count[num] += 1
        
        # 遍历出现的数字
        i = 0
        for num in range(ma + 1):
            for _ in range(count[num]):
                nums[i] = num
                i += 1
        return nums
    
    def counting_sort2(self, nums) -> None:
        """计数排序"""
        # 完整实现，可排序对象，并且是稳定排序
        # 1. 统计数组最大元素 m
        m = max(nums)
        # 2. 统计各数字的出现次数
        # counter[num] 代表 num 的出现次数
        counter = [0] * (m + 1)
        for num in nums:
            counter[num] += 1
        # 3. 求 counter 的前缀和，将“出现次数”转换为“尾索引”
        # 即 counter[num]-1 是 num 在 res 中最后一次出现的索引
        for i in range(m):
            counter[i + 1] += counter[i]
        # 4. 倒序遍历 nums ，将各元素填入结果数组 res
        # 初始化数组 res 用于记录结果
        n = len(nums)
        res = [0] * n
        for i in range(n - 1, -1, -1):
            num = nums[i]
            res[counter[num] - 1] = num  # 将 num 放置到对应索引处
            counter[num] -= 1  # 令前缀和自减 1 ，得到下次放置 num 的索引
        # 使用结果数组 res 覆盖原数组 nums
        for i in range(n):
            nums[i] = res[i]
        return nums


if __name__ == "__main__":
    print(Solution().counting_sort([49, 99, 82, 9, 57, 43, 91, 75, 15, 37]))
    print(Solution().counting_sort2([49, 99, 82, 9, 57, 43, 91, 75, 15, 37]))
