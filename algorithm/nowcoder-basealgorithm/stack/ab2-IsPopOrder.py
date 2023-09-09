# -*- coding: utf-8 -*-
# @Time    : 2023/8/15
from typing import List


class Solution:
    def IsPopOrder(self, pushV: List[int], popV: List[int]) -> bool:
        n, j = 0, 0
        for num in pushV:
            pushV[n] = num
            while n >= 0 and pushV[n] == popV[j]:
                n -= 1
                j += 1
            n += 1
        return True if n == 0 else False
    
    def IsPopOrder2(self, pushV: List[int], popV: List[int]) -> bool:
        '''辅助栈'''
        s = []
        n, j = len(pushV), 0
        for i in range(n):
            while i < n and (len(s) == 0 or s[-1] != popV[i]):
                s.append(pushV[j])
                j += 1
            if s[-1] == popV[i]:
                s.pop()
            else:
                return False
        return True


if __name__ == "__main__":
    print(Solution().IsPopOrder([1, 2, 3, 4, 5], [4, 5, 3, 2, 1]))
    print(Solution().IsPopOrder2([1, 2, 3, 4, 5], [4, 5, 3, 2, 1]))
