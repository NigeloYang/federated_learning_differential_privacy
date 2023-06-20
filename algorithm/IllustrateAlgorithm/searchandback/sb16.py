# -*- coding: utf-8 -*-
# @Time    : 2023/6/20

'''剑指 Offer 64. 求 1 + 2 + … + n
求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
'''


class Solution:
    # def sumNums(self, n: int) -> int:
    #     return n + (n >= 1 and self.sumNums(n - 1))
        # return int((1 + n) * n / 2)
        
    # 作者：Krahets
    def __init__(self):
        self.res = 0

    def sumNums(self, n: int) -> int:
        n > 1 and self.sumNums(n - 1)
        self.res += n
        return self.res



if __name__ == "__main__":
    print(Solution().sumNums(9))
