# -*- coding: utf-8 -*-
# @Time    : 2023/11/6
from typing import List


class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        if len(nums) < 3:
            return 0
        if len(nums) == 3:
            return sum(nums)
        
        nums.sort()
        min_diff = float('inf')
        n = len(nums)
        ans = 0
        
        for i in range(n - 2):
            cur = nums[i]
            l, r = i + 1, n - 1
            
            # 优化内容
            if i and cur == nums[i - 1]:
                continue  # 优化三

            s = cur + nums[i + 1] + nums[i + 2]
            if s > target:  # 后面无论怎么选，选出的三个数的和不会比 s 还小
                if s - target < min_diff:
                    ans = s  # 由于下一行直接 break，这里无需更新 min_diff
                break

            s = cur + nums[-2] + nums[-1]
            if s < target:  # cur 加上后面任意两个数都不超过 s，所以下面的双指针就不需要跑了
                if target - s < min_diff:
                    min_diff = target - s
                    ans = s
                continue
            
            
            while l < r:
                s = cur + nums[l] + nums[r]
                if s == target:
                    return s
                if s > target:
                    if s - target < min_diff:
                        min_diff = s - target
                        ans = s
                    r -= 1
                else:
                    if target - s < min_diff:
                        min_diff = target - s
                        ans = s
                    l += 1
        return ans


if __name__ == "__main__":
    print(Solution().threeSumClosest([-1, 2, 1, -4], 1))
