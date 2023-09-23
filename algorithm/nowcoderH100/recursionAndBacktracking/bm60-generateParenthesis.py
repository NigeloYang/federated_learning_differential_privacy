# -*- coding: utf-8 -*-
# @Time    : 2023/9/23


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        curstr = ''
        
        def dfs(curstr, left, right, n):
            if left == n and right == n:
                res.append(curstr)
                return
            
            if left < right:
                return
            
            if left < n:
                dfs(curstr + '(', left + 1, right, n)
            
            if right < n:
                dfs(curstr + ')', left, right + 1, n)
        
        dfs(curstr, 0, 0, n)
        return res
    
if __name__ == "__main__":
    print()
