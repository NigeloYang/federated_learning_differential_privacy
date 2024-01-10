# -*- coding: utf-8 -*-
# @Time    : 2023/10/29
from typing import List


class Solution:
    def partition(self, s: str) -> List[List[str]]:
        if not s:
            return []
        path = []
        res = []
        
        def isp(s, l, r):
            while l < r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True
        
        def dfs(s, cur):
            if cur >= len(s):                   # 如果已经遍历完了整个字符串s，则说明找到了一组回文串组合
                res.append(path[:])
                return
            
            for i in range(cur, len(s)):        # 枚举以字符s[u]为起点的所有可能的回文串
                if isp(s, cur, i):              # 如果s[u:i+1]是回文串，则将其添加进当前回文串组path中
                    path.append(s[cur: i + 1])
                else:
                    continue
                dfs(s, i + 1)    # 递归处理字符串s中剩下的部分
                path.pop()       # 回溯，将刚才添加到path中的回文串移除
        
        dfs(s, 0)
        return res


if __name__ == "__main__":
    print(Solution().partition("xxy"))
    print(Solution().partition("aaa"))
    print(123)
