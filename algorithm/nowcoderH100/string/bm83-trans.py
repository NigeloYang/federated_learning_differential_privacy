# -*- coding: utf-8 -*-
# @Time    : 2023/9/27

class Solution:
    def trans(self , s: str, n: int) -> str:
        if n <= 1:
            return s
        res = ''

        for i in range(n):
            if ord(s[i]) >= ord('a') and ord(s[i]) <= ord('z'):
                res += s[i].upper()
            elif ord(s[i]) >= ord('A') and ord(s[i]) <= ord('Z'):
                res += s[i].lower()
            elif s[i] == ' ':
                res += s[i]
        res = list(res.split(' '))[::-1]
        return ' '.join(res)
    
if __name__ == "__main__":
    print()
