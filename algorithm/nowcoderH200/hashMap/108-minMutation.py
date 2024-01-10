# -*- coding: utf-8 -*-
# @Time    : 2023/11/1
from collections import deque
from typing import List


class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        if start == end:
            return 0
        bank = set(bank)
        if end not in bank:
            return -1
        q = deque([(start, 0)])
        while q:
            cur, step = q.popleft()
            for i, x in enumerate(cur):
                for y in "ACGT":
                    if y != x:
                        nxt = cur[:i] + y + cur[i + 1:]
                        if nxt in bank:
                            if nxt == end:
                                return step + 1
                            bank.remove(nxt)
                            q.append((nxt, step + 1))
        return -1


if __name__ == "__main__":
    print(Solution().minMutation("AAAAAAAA", "CCCCCCCC",
                                 ["AAAAAAAA", "AAAAAAAC", "AAAAAACC", "AAAAAACA", "AAAAACCC", "AAAACCCC", "AAACCCCC",
                                  "AACCCCCC", "ACCCCCCC", "CCCCCCCC"]))
    print(3 // 5)
