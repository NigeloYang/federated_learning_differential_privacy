# -*- coding: utf-8 -*-
# @Time    : 2023/6/21
'''剑指 Offer 51. 数组中的逆序对
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
'''
from typing import List


class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        # 超出时间限制
        # count = 0
        # length = len(nums)
        # for fir in range(length - 1):
        #     for sec in range(fir + 1, length):
        #         if nums[fir] > nums[sec]:
        #             count += 1
        #
        # return count
        
        #
        def merge_sort(l, r):
            # 终止条件
            if l >= r:
                return 0
            # 递归划分
            m = (l + r) // 2
            res = merge_sort(l, m) + merge_sort(m + 1, r)
            # 合并阶段
            i, j = l, m + 1
            tmp[l:r + 1] = nums[l:r + 1]
            for k in range(l, r + 1):
                if i == m + 1:
                    nums[k] = tmp[j]
                    j += 1
                elif j == r + 1 or tmp[i] <= tmp[j]:
                    nums[k] = tmp[i]
                    i += 1
                else:
                    nums[k] = tmp[j]
                    j += 1
                    res += m - i + 1  # 统计逆序对
            return res
        
        tmp = [0] * len(nums)
        return merge_sort(0, len(nums) - 1)


if __name__ == "__main__":
    print(Solution().reversePairs([7, 5, 6, 4]))
    print(Solution().reversePairs([7]))
