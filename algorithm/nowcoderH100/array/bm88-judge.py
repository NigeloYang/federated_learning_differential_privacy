# -*- coding: utf-8 -*-
# @Time    : 2023/9/27

class Solution:
    def judge(self, str: str) -> bool:
        if not str:
            return
        p1, p2 = 0, len(str) - 1
        while p1 < p2:
            if str[p1] == str[p2]:
                p1 += 1
                p2 -= 1
            else:
                return False
        return True


if __name__ == "__main__":
    print(Solution().judge("absba"))
    print(Solution().judge("abaa"))
