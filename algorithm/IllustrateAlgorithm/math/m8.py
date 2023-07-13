# -*- coding: utf-8 -*-
# @Time    : 2023/7/3
'''剑指 Offer 66. 构建乘积数组
给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
'''


class Solution:
    def constructArr(self, a):
        n = len(a)
        b = []
        for i in range(n):
            temp = 1
            for j in range(n):
                if i != j:
                    temp *= a[j]
            b.append(temp)
        return b


if __name__ == "__main__":
    print(Solution().constructArr([1, 2, 3, 4, 5]))
