# -*- coding: utf-8 -*-
# @Time    : 2023/6/26

'''剑指 Offer 15. 二进制中 1 的个数
编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为 汉明重量).）。
'''


class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        for i in range(32):
            if n & 0x0001:
                count += 1
            n >>= 1
        return count
    
    def hammingWeight2(self, n: int) -> int:
        return bin(n).count('1')


if __name__ == "__main__":
    print(Solution().hammingWeight(11))
    print(Solution().hammingWeight2(11))
