# -*- coding: utf-8 -*-
# @Time    : 2023/9/18

class Solution:
    def Nqueen(self, n: int) -> int:
        states = [['#'] * n for i in range(n)]
        cset = [False] * n
        zdiag = [False] * (2 * n - 1)
        fdiag = [False] * (2 * n - 1)
        res = []
        self.backtracking(0, n, states, res, cset, zdiag, fdiag)
        
        return len(res)
    
    def backtracking(self, row, n, states, res, cset, zdiags, fdiags):
        if row == n:
            res.append([list(row) for row in states])
            return
        
        for col in range(n):
            zdiag = row - col + n - 1
            fdiag = row + col
            if not cset[col] and not zdiags[zdiag] and not fdiags[fdiag]:
                states[row][col] = 'Q'
                cset[col] = zdiags[zdiag] = fdiags[fdiag] = True
                self.backtracking(row+1,n,states,res,cset,zdiags,fdiags)
                states[row][col] = '#'
                cset[col] = zdiags[zdiag] = fdiags[fdiag] = False

if __name__ == "__main__":
    print(Solution().Nqueen(4))
