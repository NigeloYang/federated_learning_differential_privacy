# -*- coding: utf-8 -*-
# @Time    : 2023/6/21

'''剑指 Offer 33. 二叉搜索树的后序遍历序列
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同
'''
from typing import List


class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        def dfs(start, root):
            if start >= root:
                return True
            p = start
            while postorder[p] < postorder[root]:
                p += 1
            mid = p
            while postorder[p] > postorder[root]:
                p += 1
            return p == root and dfs(start, mid - 1) and dfs(mid, root - 1)
        
        return dfs(0, len(postorder) - 1)


if __name__ == "__main__":
    print(Solution().verifyPostorder([1, 6, 3, 2, 5]))
    print(Solution().verifyPostorder([1, 3, 2, 6, 5]))
    print(Solution().verifyPostorder([4, 6, 7, 5]))
