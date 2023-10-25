# -*- coding: utf-8 -*-
# @Time    : 2023/9/25


class Solution:
    def longestValidParentheses(self, s: str) -> int:
        maxlen = 0
        stack = [-1]
        
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if len(stack):
                    curlen = i - stack[len(stack) - 1]
                    maxlen = max(maxlen, curlen)
                else:
                    stack.append(i)
        return maxlen
    
    def longestValidParentheses2(self, s: str) -> int:
        maxlen = 0
        dp = [0] * len(s)
        for i in range(len(s)):
            if s[i] == ')':
                if s[i - 1] == '(':
                    dp[i] = dp[i - 2] + 2 if i >= 2 else 2
                elif (i - dp[i - 1]) > 0 and s[i - dp[i - 1] - 1] == '(':
                    dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2 if i - dp[i - 1] >= 2 else dp[i - 1] + 2
                maxlen = max(maxlen, dp[i])
        return maxlen


if __name__ == "__main__":
    print(Solution().longestValidParentheses("()(()))())()"))
    print(Solution().longestValidParentheses2("()(()))())()"))
