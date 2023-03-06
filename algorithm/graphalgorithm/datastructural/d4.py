#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/26 10:28
# @File    : d4.py
# @Author  : Richard Yang

class Solution:
    def isNumber(self, s: str) -> bool:
        states = [
            {' ': 0, 's': 1, 'd': 2, '.': 4},  # 0. start with 'blank'
            {'d': 2, '.': 4},  # 1. 'sign' before 'e'
            {'d': 2, '.': 3, 'e': 5, ' ': 8},  # 2. 'digit' before 'dot'
            {'d': 3, 'e': 5, ' ': 8},  # 3. 'digit' after 'dot'
            {'d': 3},  # 4. 'digit' after 'dot' (‘blank’ before 'dot')
            {'s': 6, 'd': 7},  # 5. 'e'
            {'d': 7},  # 6. 'sign' after 'e'
            {'d': 7, ' ': 8},  # 7. 'digit' after 'e'
            {' ': 8}  # 8. end with 'blank'
        ]
        p = 0  # start with state 0
        for c in s:
            if '0' <= c <= '9':
                t = 'd'  # digit
            elif c in "+-":
                t = 's'  # sign
            elif c in "eE":
                t = 'e'  # e or E
            elif c in ". ":
                t = c  # dot, blank
            else:
                t = '?'  # unknown
            if t not in states[p]:
                return False
            p = states[p][t]
        return p in (2, 3, 7, 8)


if __name__ == "__main__":
    print(Solution().isNumber('-+12'))
    print(Solution().isNumber('-1.2'))
    print(Solution().isNumber('-1.e2'))
    print(Solution().isNumber('-1e2'))
    data = [1,2,3,4,5,6]
    print(data[::-1])
