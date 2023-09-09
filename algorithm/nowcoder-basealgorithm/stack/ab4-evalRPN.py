# -*- coding: utf-8 -*-
# @Time    : 2023/8/15
from typing import List


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        if not tokens:
            return
        r1,r2 = 0,0
        for i in tokens:
            if tokens

if __name__ == "__main__":
    print(Solution().evalRPN(["2","1","+","4","*"]))
