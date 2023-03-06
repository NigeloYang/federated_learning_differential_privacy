'''爬楼梯
假设你正在爬楼梯。需要 n阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

示例 1：
输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶

示例 2：
输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
'''
import math


class Solution:
    def climbStairs(self, n):
        if n <= 1:
            return 1
        elif n == 2:
            return 2
        else:
            r1 = 1
            r2 = 2
            res = 0
            for i in range(3, n + 1):
                res = r1 + r2
                r1 = r2
                r2 = res
            return res


#         第二种方式
#         sqrt = math.sqrt(5)
#         return int((math.pow((1 + sqrt) / 2, n + 1) - math.pow((1 - sqrt) / 2, n + 1)) / sqrt)


if __name__ == '__main__':
    print(Solution().climbStairs(3))
    print(Solution().climbStairs(4))
    print(Solution().climbStairs(5))
