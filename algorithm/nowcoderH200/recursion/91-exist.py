# -*- coding: utf-8 -*-
# @Time    : 2023/10/30

'''
农场里的牛犊子们喜欢唱儿童谣，有一天，小牛们在一个 m x n 的二维字符网格 board 上写下了一个儿童谣，每个格子上都写了一个字母。农场主人想知道他们是否写下了一个特定的儿童谣 word。如果 word 存在于网格中，请返回 true；否则，返回 false。

儿童谣必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
'''
from typing import List
import collections


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board or len(board) == 0 or len(board[0]) == 0:
            return False
        visit = [[False] * len(board[i]) for i in range(len(board))]
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board, i, j, word, 0, visit):
                    return True
        return False
    
    def dfs(self, board: List[List[str]], i: int, j: int, word: str, wi: int, visit: List[List[bool]]) -> bool:
        if wi == len(word):
            return True
        
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[wi] or visit[i][j]:
            return False
        
        visit[i][j] = True
        if self.dfs(board, i - 1, j, word, wi + 1, visit) or self.dfs(board, i + 1, j, word, wi + 1, visit) \
            or self.dfs(board, i, j - 1, word, wi + 1, visit) or self.dfs(board, i, j + 1, word, wi + 1, visit):
            return True
        visit[i][j] = False
        return False
    
    def exist2(self, board: List[List[str]], word: str) -> bool:
        if not board or len(board) == 0 or len(board[0]) == 0:
            return False
        visit = [[False] * len(board[i]) for i in range(len(board))]
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    q = collections.deque([(i, j)])
                    visited = collections.deque()
                    wi = 0
                    while q:
                        if wi == len(word)-1:
                            return True
                        c_i, c_j = q.popleft()
                        if c_i >= 0 and c_i < len(board) and c_j >= 0 and c_j < len(board[0]) and board[c_i][c_j] == word[wi] and not visit[i][j]:
                            # print(q)
                            # print(wi)
                            # print(c_i,c_j)
                            # print(board[c_i][c_j] == word[wi])
                            # print('--------'*20)
                            wi += 1
                            visit[c_i][c_j] = True
                            visited.append((c_i, c_j))
                            for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                                n_i, n_j = c_i + di, c_j + dj
                                q.append((n_i, n_j))

                        # if c_i < 0 or c_i >= len(board) or c_j < 0 or c_j >= len(board[0]) or board[c_i][c_j] != word[wi] or visit[i][j]:
                        #     continue
                        
                        # wi += 1
                        # visit[c_i][c_j] = True
                        # visited.append((c_i, c_j))
                        # for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                        #     n_i, n_j = c_i + di, c_j + dj
                        #     q.append((n_i, n_j))

                    for vi, vj in visited:
                        visit[vi][vj] = False
        return False


if __name__ == "__main__":
    # print(Solution().exist([['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], "SABCFE"))
    # print(Solution().exist([['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], "SABCE"))
    # print(Solution().exist2([['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], "SABCFE"))
    print(Solution().exist2([['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], "SABCE"))
