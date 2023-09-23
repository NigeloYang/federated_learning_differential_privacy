# -*- coding: utf-8 -*-
# @Time    : 2023/9/13

class Solution:
    def jumpFloor(self, number: int) -> int:
        if number == 1:
            return 1
        if number == 2:
            return 2
        
        s1, s2 = 1, 2
        for i in range(2, number):
            s1, s2 = s2, s1 + s2
        
        return s2

if __name__ == "__main__":
    print()
