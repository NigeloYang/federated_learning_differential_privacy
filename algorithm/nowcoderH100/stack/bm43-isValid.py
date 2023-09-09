# -*- coding: utf-8 -*-
# @Time    : 2023/9/9

class Solution:
    def isValid(self, s: str) -> bool:
        if not s:
            return False
        
        stac = []
        
        for i in s:
            if i == '(':
                stac.append(')')
            elif i == '[':
                stac.append(']')
            elif i == '{':
                stac.append('}')
            elif stac[-1] == i:
                stac.pop()
        
        return len(stac) == 0

if __name__ == "__main__":
    print()
