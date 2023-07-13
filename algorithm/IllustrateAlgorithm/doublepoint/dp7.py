# -*- coding: utf-8 -*-
# @Time    : 2023/6/26

class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip()
        fast, low = len(s) - 1, len(s) - 1
        res = []
        while fast >= 0:
            while fast >= 0 and s[fast] != ' ':
                fast -= 1
            res.append(s[fast + 1:low + 1])
            while fast >= 0 and s[fast] == ' ':
                fast -= 1
            low = fast
        return ' '.join(res)
    def reverseWords2(self, s: str) -> str:
        return ' '.join(s.strip().split(' ')[::-1])

if __name__ == "__main__":
    print(Solution().reverseWords("the sky is blue"))
    print(Solution().reverseWords2("the sky is blue"))
    print(Solution().reverseWords(" hello world!  "))
    print(Solution().reverseWords2(" hello world!  "))
    print(" hello world!  ".strip())
