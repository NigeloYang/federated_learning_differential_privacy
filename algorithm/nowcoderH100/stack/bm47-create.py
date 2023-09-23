# -*- coding: utf-8 -*-
# @Time    : 2023/9/10

class Solution:
    def __init__(self):
        self.res = []

    def Insert(self, num):
        if len(self.res) == 0:
            self.res.append(num)
        else:
            i = 0
            while i < len(self.res):
                if num <= self.res[i]:
                   break
                i = i + 1
            self.res.insert(i, num)

    def GetMedian(self):
        n = len(self.res)
        if n % 2 == 1:
            return self.res[n // 2]
        else:
            return (self.res[n // 2] + self.res[n // 2 - 1]) / 2.0
        
if __name__ == "__main__":
    print()
