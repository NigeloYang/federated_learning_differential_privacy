# -*- coding: utf-8 -*-
# @Time    : 2023/6/20
'''剑指 Offer 16. 数值的整数次方
实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。'''


class Solution:
    def myPow(self, x: float, n: int) -> float:
        # 该方案会超时
        # res = 1.0
        # if n < 0:
        #     for i in range(abs(n)):
        #         res /= x
        # elif n == 0:
        #     return res
        # else:
        #     for i in range(n):
        #         res *= x
        # return res
        
        # 作者：Krahets
        if x == 0.0:
            return 0.0
        res = 1
        if n < 0:
            x, n = 1 / x, -n
        while n:
            if n & 1:
                res *= x
            x *= x
            n >>= 1
        return res


if __name__ == "__main__":
    print(Solution().myPow(2.0, 10))
    print(Solution().myPow(2.1, 3))
    print(Solution().myPow(2.0, -2))
    print(Solution().myPow(-2.0, -2))
    print(Solution().myPow(-2.0, -3))
