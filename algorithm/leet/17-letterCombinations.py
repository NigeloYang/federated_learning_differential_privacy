# -*- coding: utf-8 -*-
# @Time    : 2023/11/6
from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return list()
        
        phoneMap = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }
        res = []
        subres = []
        
        def backtracking(cur):
            if cur == len(digits):
                res.append(''.join(subres))
                return
            else:
                phone = digits[cur]
                for i in phoneMap[phone]:
                    subres.append(i)
                    backtracking(cur + 1)
                    subres.pop()
        backtracking(0)
        return res
    
if __name__ == "__main__":
    print(Solution().letterCombinations('23'))
