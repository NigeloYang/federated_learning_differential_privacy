'''有效的括号
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
有效字符串需满足：
左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。

示例 1：
输入：s = "()"
输出：true

示例2：
输入：s = "()[]{}"
输出：true

示例3：
输入：s = "(]"
输出：false

提示：

1 <= s.length <= 104
s 仅由括号 '()[]{}' 组成
'''


class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        dict = {')': '(', ']': '[', '}': '{'}
        for i in s:
            if i in dict and stack and stack[-1] == dict[i]:
                stack.pop()
            else:
                stack.append(i)
        return True if len(stack) == 0 else False


if __name__ == '__main__':
    print(Solution().isValid('[]'))
    print(Solution().isValid('[)'))
