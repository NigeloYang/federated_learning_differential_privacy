# -*- coding: utf-8 -*-
# @Time    : 2023/9/7

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
#
# @param root TreeNode类
# @return bool布尔型
#
class Solution:
    def isCompleteTree(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        curnodes = [root]
        
        # 标记叶节点
        flag = False
        
        while curnodes:
            nextnodes = []
            for node in curnodes:
                if not node:
                    flag = True
                else:
                    if flag:
                        return False
                    nextnodes.append(node.left)
                    nextnodes.append(node.right)
            curnodes = nextnodes
        
        return True
    
if __name__ == "__main__":
    print()
