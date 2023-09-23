# -*- coding: utf-8 -*-
# @Time    : 2023/9/16


class Solution:
    def MoreThanHalfNum_Solution(self, numbers: List[int]) -> int:
        n = len(numbers)
        dic = dict()
        
        for number in numbers:
            if number in dic:
                dic[number] += 1
            else:
                dic[number] = 1
        
        for k, v in dic.items():
            if v > n / 2:
                return k
        
        return
    
if __name__ == "__main__":
    print()
