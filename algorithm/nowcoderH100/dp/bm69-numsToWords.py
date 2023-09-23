# -*- coding: utf-8 -*-
# @Time    : 2023/9/14

class Solution:
    def solve(self , nums: str) -> int:
        if nums == '0':
            return 0
        if nums == '10' or nums == '20':
            return 1
        for i in range(1,len(nums)):
            if nums[i] == '0':
                if nums[i-1] != '1' or nums[i-1] != '2':
                    return 0
        dp = [1] * (len(nums)+1)
        for i in range(2, len(nums) + 1):
            if (nums[i - 2] == '1' and nums[i - 1] != '0') or (nums[i - 2] == '2' and nums[i - 1] > '0' and nums[i - 1] < '7'):
                dp[i] = dp[i - 1] + dp[i - 2]
            else:
                dp[i] = dp[i - 1]
        return dp[len(nums)]

    def solve2(self, nums: str) -> int:
        if nums == '0':
            return 0
        dp = [0] * len(nums)
        dp[0] = 1
        for i in range(1,len(nums)):
            if nums[i] == '0':
                if nums[i-1] == '0' or nums[i-1] > '2':
                    return 0
            else:
                dp[i] = dp[i-1]
                
            if '10' <= nums[i - 1] + nums[i] < '27':
                if i == 1:
                    dp[i] += 1
                else:
                    dp[i] += dp[i-2]
        return dp[len(nums)-1]
    
if __name__ == "__main__":
    print(Solution().solve("0"))
    print(Solution().solve("100"))
    print(Solution().solve("12"))
    print(Solution().solve("160"))
    print(Solution().solve("72910721221427251718216239162221131917242"))
    print(Solution().solve("72416145196211821232022471311148103136128331523141061051992231223"))
    print('----'*20)
    print(Solution().solve2("0"))
    print(Solution().solve2("12"))
    print(Solution().solve2("100"))
    print(Solution().solve2("160"))
    print(Solution().solve2("72910721221427251718216239162221131917242"))
    print(Solution().solve2("72416145196211821232022471311148103136128331523141061051992231223"))
