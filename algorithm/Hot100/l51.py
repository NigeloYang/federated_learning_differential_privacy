# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''


'''


class Solution:
    def solveNQueens(self, n: int):
        def backtrack(row, n, state, res, cols, diags1, diags2):
            # 记录所有结果
            if row == n:
                res.append([list(row) for row in state])
                return
            
            # 遍历所有列
            for col in range(n):
                # 计算当前位置皇后对应的主副对角线索引
                diag1 = row - col + n - 1
                diag2 = row + col
                
                # 剪枝：不允许该格子所在列、主对角线、副对角线存在皇后
                if not cols[col] and not diags1[diag1] and not diags2[diag2]:
                    state[row][col] = 'Q'
                    cols[col] = diags1[diag1] = diags2[diag2] = True
                    backtrack(row + 1, n,state, res, cols, diags1, diags2)
                    state[row][col] = '#'
                    cols[col] = diags1[diag1] = diags2[diag2] = False
        
        # 初始化棋盘
        state = [['#'] * n for _ in range(n)]
        cols = [False] * n  # 记录列是否有皇后
        diags1 = [False] * (2 * n - 1)  # 记录主对角线是否有皇后
        diags2 = [False] * (2 * n - 1)  # 记录副对角线是否有皇后
        res = []
        backtrack(0, n, state, res, cols, diags1, diags2)
        return res


if __name__ == "__main__":
    print(len(Solution().solveNQueens(5)))
