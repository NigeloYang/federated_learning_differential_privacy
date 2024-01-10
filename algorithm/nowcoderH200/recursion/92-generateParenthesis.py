# -*- coding: utf-8 -*-
# @Time    : 2023/10/30


'''
农场里有 n 头牛，农场主人需要给这些牛搭建一个圈形的围栏，每头牛都需要一个独立的空间。为了使得围栏更加稳定，农场主人决定使用木棍和铁链来固定围栏。每头牛的空间由一个木棍和两个铁链组成，且木棍和铁链的连接处必须是一个完整的括号。数字 n 代表牛的数量，请你设计一个函数，用于生成所有可能的并且稳定的围栏组合。
'''
from typing import List


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []  # 用于存储最终结果的列表
        
        def backtrack(sb, open, close):
            if len(sb) == 2 * n:  # 生成了一个完整的括号组合
                result.append(sb)
                return
            
            if open < n:  # 可以添加左括号
                print(sb)
                backtrack(sb + '(', open + 1, close)
            
            if close < open:  # 可以添加右括号
                print(sb)
                backtrack(sb + ')', open, close + 1)
        
        backtrack('', 0, 0)  # 回溯生成括号组合
        
        return result


if __name__ == "__main__":
    print(Solution().generateParenthesis(3))
