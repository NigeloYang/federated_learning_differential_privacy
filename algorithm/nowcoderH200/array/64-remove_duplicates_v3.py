# -*- coding: utf-8 -*-
# @Time    : 2023/10/25
from typing import List
'''
在一片广阔的草原上，有一群牛正在享受美味的青草。我们可以用一个升序排列的数组 nums 表示这群牛的位置（用整数表示）。由于草原太大，相同位置可能有多头牛。为了维护草原上的生态平衡，牧人希望我们去计算各个位置上牛群数量的有序分布，要求一个位置上的牛群不能超过 3 头。请返回重新分布后的数组长度和重新分布的数组，后者通过引用返回。
'''

class Solution:
    def remove_duplicates_v3(self, nums: List[int]) -> int:
        if not nums:
            return 0
        l, f = 0, 1
        count = 1
        while f < len(nums):
            if nums[f] == nums[f - 1]:
                if count < 3:
                    nums[l] = nums[f]
                    l += 1
                    count += 1
            else:
                nums[l] = nums[f]
                l += 1
                count = 1
            f += 1
    
        return l + 1,nums
    
if __name__ == "__main__":
    print(Solution().remove_duplicates_v3([1, 1, 1, 1, 2, 2, 2, 3]))
    print(Solution().remove_duplicates_v3([1, 1, 1, 1, 2, 3, 3, 3, 3, 3]))
