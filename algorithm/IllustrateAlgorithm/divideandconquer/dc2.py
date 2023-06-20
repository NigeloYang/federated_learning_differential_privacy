# -*- coding: utf-8 -*-
# @Time    : 2023/6/20
'''剑指 Offer 16. 数值的整数次方
实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。'''


class Solution:
    def myPow(self, x: float, n: int) -> float:
        res = 1.0
        if n < 0:
            for i in range(abs(n)):
                res /= x
        elif n == 0:
            return res
        else:
            for i in range(n):
                res *= x
        return res

if __name__ == "__main__":
    print(Solution().myPow(2.0,10))
    print(Solution().myPow(2.1,3))
    print(Solution().myPow(2.0,-2))
    print(Solution().myPow(-2.0,-2))
    print(Solution().myPow(-2.0,-3))
