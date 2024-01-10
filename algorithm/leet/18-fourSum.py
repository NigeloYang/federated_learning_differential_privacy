# -*- coding: utf-8 -*-
# @Time    : 2023/11/6
from typing import List


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        if not nums or len(nums) < 4:
            return list()
        
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
                break
            if nums[i] + nums[-3] + nums[-2] + nums[-1] < target:
                continue
            for j in range(i + 1, n - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                if nums[i] + nums[j] + nums[i + 1] + nums[i + 2] > target:
                    break
                if nums[i] + nums[j] + nums[-2] + nums[-1] < target:
                    continue
                l, r = j + 1, n - 1
                while l < r:
                    total = nums[i] + nums[j] + nums[l] + nums[r]
                    if total == target:
                        res.append([nums[i], nums[j], nums[l], nums[r]])
                        while l < r and nums[l] == nums[l + 1]:
                            l += 1
                        l += 1
                        while l < r and nums[r] == nums[r - 1]:
                            r -= 1
                        r -= 1
                    elif total < target:
                        l += 1
                    else:
                        r -= 1
        return res


if __name__ == "__main__":
    print(Solution().fourSum(nums=[1, 0, -1, 0, -2, 2], target=0))
    print(Solution().fourSum(nums=[2, 2, 2, 2, 2, 2], target=8))
