# -*- coding: utf-8 -*-
# @Time    : 2023/9/9

class Solution:
    def maxInWindows(self, num: List[int], size: int) -> List[int]:
        if len(num) < size or size == 0:
            return
        
        res = []
        
        for i in range(len(num) - size + 1):
            res.append(max(num[i:i + size]))
        return res

if __name__ == "__main__":
    print()
