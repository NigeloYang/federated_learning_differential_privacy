# -*- coding: utf-8 -*-
# @Time    : 2023/6/19

class Solution:
    def translateNum(self, num):
        # 方案1
        # s = str(num)
        # a, b = 1, 1
        # for i in range(2, len(s)+1):
        #     a, b = (a + b if '10' <= s[i - 2:i] <= '25' else a), a
        # return a
        
        # 方案2
        # s = str(num)
        # a, b = 1, 1
        # for i in range(2, len(s) + 1):
        #     if '10' <= s[i - 2:i] <= '25':
        #         a, b = a + b, a
        #     else:
        #         a, b = a, a
        # return a
        
        # 数字取余计算
        a = b = 1
        y = num % 10
        while num > 9:
            num //= 10
            x = num % 10
            temp = 10 * x + y
            c = a + b if 10 <= temp <= 25 else a
            a, b = c, a
            y = x
        return a


if __name__ == "__main__":
    print(Solution().translateNum(12258))
