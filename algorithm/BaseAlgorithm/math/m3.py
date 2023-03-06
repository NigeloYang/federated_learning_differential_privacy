'''3的幂
给定一个整数，写一个函数来判断它是否是 3的幂次方。如果是，返回 true ；否则，返回 false 。
整数 n 是 3 的幂次方需满足：存在整数 x 使得 n == 3x


示例 1：
输入：n = 27
输出：true
示例 2：
输入：n = 0
输出：false
示例 3：
输入：n = 9
输出：true
示例 4：
输入：n = 45
输出：false


提示：
-231 <= n <= 231 - 1
'''


class Solution:
    def isPowerOfThree(self, n):
        # return (n > 0 and 1162261467 % n == 0)
        # if n == 0:
        #     return False
        #
        # elif n % 3 == 0:
        #     for i in range(n):
        #         if 3 ** i == n:
        #             return True
        # else:
        #     return False
        count = False
        for i in range(32):
            if 3**i == n:
                return True
            else:
                count = False
        return count

if __name__ == '__main__':
    #     for i in range(31):
    #         if 3 ** i < 2 ** 31 - 1:
    #             print(i, 3 ** i)
    print(Solution().isPowerOfThree(27))
    print(Solution().isPowerOfThree(0))
    print(Solution().isPowerOfThree(9))
    print(Solution().isPowerOfThree(45))
