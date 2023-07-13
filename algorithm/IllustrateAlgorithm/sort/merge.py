# -*- coding: utf-8 -*-
# @Time    : 2023/6/21
'''归并排序体现了 “分而治之” 的算法思想，具体为：

「分」：不断将数组从 中点位置 划分开，将原数组的排序问题转化为子数组的排序问题；
「治」：划分到子数组长度为 1 时，开始向上合并，不断将 左右两个较短排序数组 合并为 一个较长排序数组，直至合并至原数组时完成排序；
'''


class Solution:
    def merge_sort(self, nums, left, right):
        #  终止条件
        if left >= right:
            return
        # 递归划分数组
        mid = (left + right) // 2
        self.merge_sort(nums, left, mid)
        self.merge_sort(nums, mid + 1, right)
        
        # 合并
        self.merge(nums, left, mid, right)
        
        return nums
    
    def merge(self, nums, left, mid, right):
        """合并左子数组和右子数组"""
        # 左子数组区间 [left, mid]
        # 右子数组区间 [mid + 1, right]
        # 初始化辅助数组
        tmp = list(nums[left: right + 1])
        print(left, mid, right, tmp)
        # 左子数组的起始索引和结束索引
        left_start = 0
        left_end = mid - left
        # 右子数组的起始索引和结束索引
        right_start = mid + 1 - left
        right_end = right - left
        
        # 通过覆盖原数组 nums 来合并左子数组和右子数组
        for k in range(left, right + 1):
            # 若“左子数组已全部合并完”，则选取右子数组元素，并且 j++
            if left_start > left_end:
                nums[k] = tmp[right_start]
                right_start += 1
            # 否则，若“右子数组已全部合并完”或“左子数组元素 <= 右子数组元素”，则选取左子数组元素，并且 i++
            elif right_start > right_end or tmp[left_start] <= tmp[right_start]:
                nums[k] = tmp[left_start]
                left_start += 1
            # 否则，若“左右子数组都未全部合并完”且“左子数组元素 > 右子数组元素”，则选取右子数组元素，并且 j++
            else:
                nums[k] = tmp[right_start]
                right_start += 1


if __name__ == "__main__":
    data1 = [1, 3, 2, 1, 5, 2]
    data2 = [1, 3, 2, 7, 5, 2, 11, 10]
    print(Solution().merge_sort(data2, 0, len(data2) - 1))
    print(Solution().merge_sort(data1, 0, len(data1) - 1))

