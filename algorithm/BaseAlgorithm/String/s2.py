'''给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。

如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。

假设环境不允许存储 64 位整数（有符号或无符号）。
 

示例 1：

输入：x = 123
输出：321
示例 2：

输入：x = -123
输出：-321
示例 3：

输入：x = 120
输出：21
示例 4：

输入：x = 0
输出：0
'''


def reverse(x):
    # 方法1
    # s = str(x)
    # if s[0] == '-':
    #     s = s[0] + s[-1:-len(s):-1]
    # else:
    #     s = s[::-1]
    # s = int(s)
    # if -2 ** 31 <= s <= 2 ** 31 - 1:
    #     return s
    # return 0

    # 方法2
    res = 0
    op = 1 if x > 0 else -1
    x = abs(x)
    while not x == 0:
        res = int(res * 10 + x % 10)
        x = int(x / 10)
    if -2 ** 31 <= res <= 2 ** 31 - 1:
        return op * res
    return 0


print(reverse(123))
print(reverse(-123))
print(reverse(120))
