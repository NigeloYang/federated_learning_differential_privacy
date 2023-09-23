# -*- coding: utf-8 -*-
# @Time    : 2023/9/23

class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        num = 0
        sign = "+"
        i = 0
        while i < len(s):
            if s[i] == '(':
                j = i + 1
                lens = 1
                while lens > 0:
                    if s[j] == '(': lens += 1
                    if s[j] == ')': lens -= 1
                    j += 1
                num = self.calculate(s[i + 1:j - 1])
                i = j - 1
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            if s[i] in '+-*/' or i == len(s) - 1:
                if sign == "+": stack.append(num)
                elif sign == "-": stack.append(-1 * num)
                elif sign == "*": stack.append(stack.pop() * num)
                elif sign == "/": stack.append(int(stack.pop() / num))
                num = 0
                sign = s[i]
            i += 1
        return sum(stack)


if __name__ == "__main__":
    print(Solution().calculate('(3+4)*(5+(2-3))'))
    print(Solution().solve('(3+4)*(5+(2-3))'))
