# -*- coding: utf-8 -*-
# @Time    : 2023/6/19

'''剑指 Offer 12. 矩阵中的路径

输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
'''


class Solution:
    def exist(self, board, word):
        m = len(board)
        n = len(board[0])
        road = [[0] * n for i in range(m)]
        


if __name__ == "__main__":
    board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    word = "ABCCED"
    print(Solution().exist(board, word))
