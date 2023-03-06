import TreeNode as trees


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# class Solution:
#     def isSymmetric(self, root):
#         if root is None:
#             return True
#         queue = []
#         queue.append(root)
#         while len(queue) > 0:
#             sublist = []
#             for i in range(len(queue)):
#                 root = queue.pop(0)
#                 sublist.append(root.val)
#                 if root.left is not None:
#                     queue.append(root.left)
#                 if root.right is not None:
#                     queue.append(root.right)
#             if len(sublist) == 1:
#                 return True
#             elif len(sublist) % 2 == 0:
#                 for i in range(len(sublist) / 2):
#                     if sublist[i] != sublist[len(sublist) - 1 -i]:
#                         return False
#             else:
#                 return False
#         return True

class Solution:
    def isSymmetric(self, root):
        if root is None or not (root.left or root.right):
            return True
        queue = [root.left, root.right]
        while len(queue) > 0:
            left = queue.pop(0)
            right = queue.pop(0)
            if not (left or right):
                continue
            if not (left and right):
                return False
            if left.val != right.val:
                return False
            queue.append(left.left)
            queue.append(right.right)
            
            queue.append(left.right)
            queue.append(right.left)
        
        return True


if __name__ == "__main__":
    data = [1, 2, 2, 3, 4, 4, 3]
    data2 = [1, 2, 2, None, 3, None, 3]
    data3 = [1,
             2, 2,
             3, 4, 4, 3,
             5, 6, 7, 8, 8, 7, 6, 5]
    tree1 = trees.Tree()
    tree2 = trees.Tree()
    tree3 = trees.Tree()
    
    tree1.createTree(data)
    tree1.breadth_travel()
    print()
    tree2.createTree(data2)
    tree2.breadth_travel()
    print()
    tree3.createTree(data3)
    tree3.breadth_travel()
    
    print('\n 验证二叉搜索树结果:', Solution().isSymmetric(tree1.root))
    print('\n 验证二叉搜索树结果:', Solution().isSymmetric(tree2.root))
    print('\n 验证二叉搜索树结果:', Solution().isSymmetric(tree3.root))

