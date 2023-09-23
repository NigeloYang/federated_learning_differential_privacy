# -*- coding: utf-8 -*-
# @Time    : 2023/9/18


class Solution:
    def Permutation(self, str: str) -> List[str]:
        strs, res = list(str), []
        
        def dfs(x):
            if x == len(strs) - 1:
                res.append(''.join(strs))
                return
            dic = set()
            for i in range(x, len(strs)):
                if strs[i] in dic:
                    continue
                dic.add(strs[i])
                strs[i], strs[x] = strs[x], strs[i]
                dfs(x + 1)
                strs[i], strs[x] = strs[x], strs[i]
        
        dfs(0)
        return res


if __name__ == "__main__":
    print()
