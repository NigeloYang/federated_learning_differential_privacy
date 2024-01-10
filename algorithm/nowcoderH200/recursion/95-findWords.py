# -*- coding: utf-8 -*-
# @Time    : 2023/10/31
from typing import List

'''
农场主有一个 m x n 的二维牛群定位系统，系统通过字符网格 board 表示，每个字符代表一头牛的名字的一个字母。此外，农场主还有一个单词列表 words，表示所有牛的名字。请你找出所有在二维定位系统上出现的牛名，输出顺序按照在words中的出现顺序。
牛名必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个牛名中不允许被重复使用。
'''


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not board[0]:
            return []
        ans = {}
        for b in board:
            st = ''.join(b)
            if st in words:
                i = words.index(st)
                ans[st] = i

        

if __name__ == "__main__":
    print(Solution().findWords([['a', 'b'], ['c', 'd']], ["abcb"]))
