# -*- coding: utf-8 -*-
# @Time    : 2023/10/31
from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        ans = []
        path = []
        visited = [False] * n
        
        def dfs(cur):
            if len(path) == k:
                ans.append(path[:])
                return
            
            for i in range(cur, n):
                # if not visited[i]:
                    # visited[i] = True
                    path.append(i + 1)
                    dfs(i + 1)
                    path.pop()
                    # visited[i] = False

        dfs(0)
        return ans


if __name__ == "__main__":
    print(Solution().combine(5, 3))
