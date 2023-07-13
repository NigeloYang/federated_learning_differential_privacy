# -*- coding: utf-8 -*-
# @Time    : 2023/7/3
'''剑指 Offer 39. 数组中出现次数超过一半的数字
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。'''


class Solution:
    def majorityElement(self, nums) -> int:
        leng = len(nums)
        count = {}
        for num in nums:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
        
        for k, v in count.items():
            if v > (leng // 2):
                return k
        return
    
    def majorityElement2(self, nums) -> int:
        sorted(nums)
        return nums[len(nums) // 2]
    
    def majorityElement3(self, nums) -> int:
        # 摩尔投票法，两两相互抵消
        vote = 0
        for num in nums:
            if vote == 0:
                x = num
            vote += 1 if num == x else -1
        return x


if __name__ == "__main__":
    print(Solution().majorityElement([1, 2, 3, 2, 2, 2, 5, 4, 2]))
    print(Solution().majorityElement2([1, 2, 3, 2, 2, 2, 5, 4, 2]))
