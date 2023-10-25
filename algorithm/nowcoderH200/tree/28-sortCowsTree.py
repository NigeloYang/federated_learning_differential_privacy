# -*- coding: utf-8 -*-
# @Time    : 2023/10/15

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def sortCowsTree(self, cows: List[int]) -> TreeNode:
        if not cows:
            return
        
        root = TreeNode(-1)
        leftnums = [root]
        rightnums = [root]
        
        for i in range(len(cows)):
            if cows[i] == 0:
                node = TreeNode(cows[i])
                if not leftnums[0].left:
                    leftnums[0].left = node
                    if leftnums[0] == root:
                        leftnums.pop(0)
                else:
                    leftnums[0].right = node
                    leftnums.pop(0)
                leftnums.append(node)
            else:
                node = TreeNode(cows[i])
                if not rightnums[0].left and rightnums[0] != root:
                    rightnums[0].left = node
                else:
                    rightnums[0].right = node
                    rightnums.pop(0)
                rightnums.append(node)
        
        return root
    
if __name__ == "__main__":
    print()
