# -*- coding: utf-8 -*-
# @Time    : 2023/8/15

class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for i in s:
            if i == '(':
                stack.append(')')
            elif i == '[':
                stack.append(']')
            elif i == '{':
                stack.append('}')
            elif len(stack) == 0:
                return False
            elif stack[-1] == i:
                stack.pop()
        return len(stack) == 0
    
    def isValid2(self, s: str) -> bool:
        stack = []
        dict = {')': '(', ']': '[', '}': '{'}
        for i in s:
            if i in dict and stack and stack[-1] == dict[i]:
                stack.pop()
            else:
                stack.append(i)
        return len(stack) == 0


if __name__ == "__main__":
    print(Solution().isValid("[]"))
    print(Solution().isValid2("[]"))
    print(Solution().isValid2("[]["))
