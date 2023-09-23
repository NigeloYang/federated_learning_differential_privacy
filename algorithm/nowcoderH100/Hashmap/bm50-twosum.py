# -*- coding: utf-8 -*-
# @Time    : 2023/9/16


class Solution:
    def twoSum(self , numbers: List[int], target: int) -> List[int]:
        dic = dict()

        for i in range(len(numbers)):
            if target - numbers[i] in dic:
                return [dic[target - numbers[i]],i+1]
            else:
                dic[numbers[i]] = i + 1
        return []
    
if __name__ == "__main__":
    print()
