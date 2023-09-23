# -*- coding: utf-8 -*-
# @Time    : 2023/9/16

class Solution:
    def permute(self , nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res,path,used = [],[], [False]*n

        def dfs(cur):
            if cur == n:
                res.append(path[:])
                return

            for i in range(n):
                if used[i] == False:
                    used[i] = True
                    path.append(nums[i])
                    dfs(cur+1)
                    path.pop()
                    used[i] = False

        dfs(0)
        return res
    
if __name__ == "__main__":
    print()
