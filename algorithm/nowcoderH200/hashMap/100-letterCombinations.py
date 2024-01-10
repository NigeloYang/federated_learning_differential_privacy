# -*- coding: utf-8 -*-
# @Time    : 2023/10/31
from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        dic = {'2': "abc", '3': "def", '4': "ghi", '5': "jkl", '6': "mno", '7': "pqrs", '8': "tuv", '9': "wxyz"}
        if not digits:
            return []
        ans = []
        self.backtrack(digits, ans, [], dic, 0)
        return ans
    
    def backtrack(self, digits: str, ans: List[str], path: List[str], dic: dict, ind: int):
        if len(path) == len(digits):
            ans.append(''.join(path))
            return
        digit = digits[ind]
        tempS = dic[digit]
        for i in range(len(tempS)):
            path.append(tempS[i])
            self.backtrack(digits, ans, path, dic, ind + 1)
            path.pop()


if __name__ == "__main__":
    print(Solution().letterCombinations('345'))
