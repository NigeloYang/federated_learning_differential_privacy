# -*- coding: utf-8 -*-
# @Time    : 2023/6/26

'''剑指 Offer 57 - II. 和为 s 的连续正数序列
输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
'''
class Solution:
    def findContinuousSequence(self, target: int):
        # 滑动窗口（双指针）
        i, j, s, res = 1, 2, 3, []
        while i < j:
            if s == target:
                res.append(list(range(i, j+1)))
                j += 1
                s += j
            elif s < target:
                j += 1
                s += j
            else:
                s -= i
                i += 1
        return res

    def findContinuousSequence2(self, target: int):
        i, j, s, res = 1, 2, 3, []
        while i < j:
            if s == target:
                res.append(list(range(i, j + 1)))
            if s >= target:
                s -= i
                i += 1
            else:
                j += 1
                s += j
        return res


if __name__ == "__main__":
    print(Solution().findContinuousSequence(9))
    print(Solution().findContinuousSequence2(9))
    print(Solution().findContinuousSequence(12))
