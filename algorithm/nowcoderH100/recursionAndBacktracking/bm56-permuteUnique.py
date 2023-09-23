# -*- coding: utf-8 -*-
# @Time    : 2023/9/17

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dfs(x):
            if x == len(nums) - 1:
                res.append(list(nums))
                return
            dic = set()
            for i in range(x,len(nums)):
                if nums[i] in dic:
                    continue
                dic.add(nums[i])
                nums[i],nums[x] = nums[x],nums[i]
                dfs(x+1)
                nums[i],nums[x] = nums[x],nums[i]
        res = []
        dfs(0)
        return res

if __name__ == "__main__":
    print()
