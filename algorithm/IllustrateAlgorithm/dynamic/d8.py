# -*- coding: utf-8 -*-
# @Time    : 2023/6/19

class Solution:
    def nthUglyNumber(self, n):
        dp, a, b, c = [1] * n, 0, 0, 0
        for i in range(1, n):
            n2, n3, n5 = 2 * dp[a], 3 * dp[b], 5 * dp[c]
            dp[i] = min(n2, n3, n5)
            if dp[i] == n2: a += 1
            if dp[i] == n3: b += 1
            if dp[i] == n5: c += 1
        return dp[-1]


if __name__ == "__main__":
    print(Solution().nthUglyNumber(10))
