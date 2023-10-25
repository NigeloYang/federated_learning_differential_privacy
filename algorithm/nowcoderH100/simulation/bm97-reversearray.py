# -*- coding: utf-8 -*-
# @Time    : 2023/9/30

class Solution:
    def solve(self, n: int, m: int, a: List[int]) -> List[int]:
        if m == 0:
            return a
        m = m % n
        
        a.reverse()
        b = a[:m]
        b.reverse()
        a[:m] = b
        c = a[m:]
        c.reverse()
        a[m:] = c
        return a


if __name__ == "__main__":
    print()
