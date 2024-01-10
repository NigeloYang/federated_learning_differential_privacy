# -*- coding: utf-8 -*-
# @Time    : 2023/10/31
from typing import List


class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        if not arr:
            return
        
        ans = [0] * 1001
        for a in arr:
            ans[a] += 1
        
        dic = {}
        for i in range(len(arr)):
            if ans[arr[i]] > 0:
                if ans[i] in dic:
                    return False
                else:
                    dic[arr[i]] = ans[arr[i]]
        return True

    def uniqueOccurrences2(self, arr: List[int]) -> bool:
        if not arr:
            return
    
        dic = {}
        for a in arr:
            if a in dic:
                dic[a] += 1
            else:
                dic[a] = 1
        count = set()
        for v in dic.values():
            if v in count:
                return False
            else:
                count.add(v)
            
        return True
    
if __name__ == "__main__":
    print(Solution().uniqueOccurrences([500, 500, 600]))
