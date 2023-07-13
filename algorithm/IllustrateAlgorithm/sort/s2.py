# -*- coding: utf-8 -*-
# @Time    : 2023/6/22

'''剑指 Offer 41. 数据流中的中位数
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。
如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
'''


class MedianFinder:
    
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.nums = []
        self.size = 0
    
    def addNum(self, num: int) -> None:
        if num is not None:
            self.nums.append(num)
        else:
            self.nums.append(0)
        self.size += 1

    def findMedian(self) -> float:
        self.nums.sort()
        if self.size // 2 == 0:
            return (self.nums[0] + self.nums[-1]) / 2
        else:
            return self.nums[self.size // 2]


if __name__ == "__main__":
    # Your MedianFinder object will be instantiated and called as such:
    obj = MedianFinder()
    obj.addNum()
    param_2 = obj.findMedian()
