# -*- coding: utf-8 -*-
# @Time    : 2023/11/23

class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        min_int = -2 ** 31
        if dividend == min_int and divisor == -1:
            return -min_int - 1
        a, b, res = abs(dividend), abs(divisor), 0
        for i in range(31, -1, -1):
            # 2^i * b <= a 换句话说 a/b = 2^i + (a-2^i*b)/b
            if (b << i) <= a:
                res += 1 << i
                a -= b << i
        return res if (dividend > 0) == (divisor > 0) else -res
    
    def divide2(self, dividend: int, divisor: int) -> int:
        result = int(dividend / divisor)
        if result <= -2 ** 31:
            return -2 ** 31
        if result >= 2 ** 31 - 1:
            return 2 ** 31 - 1
        else:
            return result


if __name__ == "__main__":
    print(Solution().divide(10, 3))
    print(Solution().divide2(10, 3))
    print(Solution().divide(7, -3))
    print(Solution().divide2(7, -3))
