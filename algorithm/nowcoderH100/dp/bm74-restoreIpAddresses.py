# -*- coding: utf-8 -*-
# @Time    : 2023/9/23
from typing import List


class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        if len(s) < 4 and len(s) > 12:
            return []
        
        res = []
        maxnums = ''
        
        self.backtracking(s, res, maxnums, 0, 0)
        return res
    
    def backtracking(self, s, res, maxnums, step, index):
        cur = ''
        if step == 4:
            if index != len(s):
                return
            res.append(maxnums)
        else:
            i = index
            while i < index + 3 and i < len(s):
                cur += s[i]
                num = int(cur)
                temp = maxnums
                if num <= 255 and (len(cur) == 1 or cur[0] != '0'):
                    if step - 3 != 0:
                        maxnums += cur + '.'
                    else:
                        maxnums += cur
                    self.backtracking(s, res, maxnums, step + 1, i + 1)
                    maxnums = temp
                i += 1
if __name__ == "__main__":
    print(Solution().restoreIpAddresses('25525511135'))
    print(Solution().restoreIpAddresses('25522'))
