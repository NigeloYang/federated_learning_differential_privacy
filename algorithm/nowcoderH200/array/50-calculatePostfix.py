# -*- coding: utf-8 -*-
# @Time    : 2023/10/23
from typing import List


class Solution:
    def calculatePostfix(self , tokens: List[str]) -> int:
        if not tokens:
            return 0

        sta = []

        for v in tokens:
            try:
                sta.append(int(v))
            except:
                ans = 0
                if v == '+':
                    ans += sta.pop() + sta.pop()
                elif v == '-':
                    temp = sta.pop()
                    ans += sta.pop() - temp
                elif v =='*':
                    ans += sta.pop() * sta.pop()
                elif v == '/':
                    temp = sta.pop()
                    ans += int(sta.pop() / temp)
                sta.append(ans)

        return sta.pop()
    
if __name__ == "__main__":
    print(Solution().calculatePostfix(["4","13","5","/","+"]))
