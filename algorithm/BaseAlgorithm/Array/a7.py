'''加一
给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。
最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
你可以假设除了整数 0 之外，这个整数不会以零开头。

示例1：
输入：digits = [1,2,3]
输出：[1,2,4]
解释：输入数组表示数字 123。

示例2：
输入：digits = [4,3,2,1]
输出：[4,3,2,2]
解释：输入数组表示数字 4321。

示例 3：
输入：digits = [0]
输出：[1]

提示：
1 <= digits.length <= 100
0 <= digits[i] <= 9
'''


def plusOne(digits):
    # 只考虑了最后一个数字非整体考虑
    # if digits[-1] + 1 == 10:
    #     t1 = str(digits[-1] + 1)
    #     temp = [i for i in t1]
    #     digits[-1] = int(temp[0])
    #     digits.append(int(temp[1]))
    #     return digits
    # else:
    #     digits[-1] += 1

    # 整体考虑
    if digits[-1] + 1 == 10:
        res = ''.join(str(i) for i in digits)
        res = int(res) + 1
        digits = [int(i) for i in str(res)]
        return digits
    else:
        digits[-1] += 1
        return digits


print(plusOne([1, 2, 3]))
print(plusOne([4, 3, 2, 1]))
print(plusOne([0]))
print(plusOne([1,9]))
print(plusOne([9,9]))
